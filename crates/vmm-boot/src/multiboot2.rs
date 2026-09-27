// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Multiboot2 direct boot, for kernels such as illumos `unix`.
//!
//! The kernel is entered in 32-bit protected mode with paging off, `EAX`
//! holding the Multiboot2 bootloader magic and `EBX` the address of the
//! boot information structure (MBI). The image is placed with the header's
//! address tag, so an ELF64 kernel whose loadable code is 32-bit (illumos
//! dboot) boots without the VMM interpreting its program headers.
//!
//! The optional initrd becomes the single module, tagged `type=rootfs`,
//! which is what illumos dboot expects of a boot archive. Boot properties
//! travel on the command line (`-B name=value,...`).
//!
//! # Guest memory layout
//!
//! ```text
//! 0x0000_0000  real-mode IVT, every vector pointing into the BIOS shim
//! 0x0000_1000  MBI (tags, up to MB2_MBI_MAX bytes)
//! 0x0000_6000  GDT (null + 32-bit code + 32-bit data)
//! <from tag>   kernel image, header address tag load_addr ..
//! 0x000F_8000  legacy BIOS shim (real-mode handlers)
//! 0x000F_8040  firmware prologue (32-bit), the BSP's first instruction
//! 0x000F_FFF5  BIOS date string "mm/dd/yy" and model byte at 0xFFFFE
//! <above image> module (page-aligned, directly above the kernel)
//! ```
//!
//! # Legacy BIOS shim
//!
//! A Multiboot2 boot without EFI tags tells illumos it runs on a legacy
//! BIOS, and the kernel then issues a few real-mode `int` calls through
//! its own trampoline: `int 1Ah` for PCI BIOS detection and IRQ routing,
//! `int 10h` to set the console mode and `int 13h` for BIOS disk data.
//! There is no firmware behind a direct boot, so the VMM installs a tiny
//! handler set: PCI BIOS present (`AX=B101h`) reports configuration
//! mechanism 1 on bus 0, and every other call returns carry set
//! (function not supported), which each caller already handles.
//!
//! # BIOS date
//!
//! On a boot without EFI tags illumos decides whether ACPI exists by reading
//! the BIOS ROM date at 0xFFFF5 (`acpica_check_bios_date`) and accepts only
//! 1999 or later. With no ROM behind the address it turns ACPI off, both APIC
//! platform modules fail to initialise ACPICA, and the kernel falls back to
//! the uniprocessor 8259 module: no SMP, no MSI/MSI-X, and legacy interrupts
//! nobody routes. The VMM therefore writes a date string and the AT model byte
//! there, like any BIOS ROM. (`-B acpi-user-options=0x1` is the guest-side
//! workaround.)
//!
//! # Firmware prologue
//!
//! Firmware leaves PCI devices with I/O, memory and bus-master decoding
//! enabled. illumos drivers on a conventional PCI bus rely on that and do
//! not set bus mastering themselves, so without firmware their DMA never
//! reaches memory (vioblk waits forever, nvme I/O fails with EIO). The BSP
//! therefore starts in a short 32-bit prologue that sets those three
//! command bits on every present function 0 of bus 0 through configuration
//! mechanism 1, restores `EAX` and jumps to the kernel entry.

use anyhow::{bail, ensure, Context, Result};

use vmm_core::mem::MemCtx;
use vmm_devices::acpi;

use crate::bsp::{
    self, FlatSegments, CR0_ET, CR0_PE, GDT_DATA, RFLAGS_RESERVED_BIT1,
    SEG_TYPE_CODE_XRA,
};
use crate::direct::CMDLINE_MAX;
use crate::{BootVcpu, ImageBytes, InitrdImage};

// ── Guest physical addresses ────────────────────────────────────────

const MB2_GPA_MBI: u64 = 0x1000;
/// The MBI must end before the GDT.
const MB2_MBI_MAX: usize = 0x5000;
const MB2_GPA_GDT: u64 = 0x6000;
/// Real-mode segment and physical address of the BIOS shim. Clear of the
/// SMBIOS entry point and tables (0xF0000, 0xF1000) and the ACPI tables
/// (0xF2400).
const BIOS_SHIM_SEG: u16 = 0xF800;
const BIOS_SHIM_GPA: u64 = (BIOS_SHIM_SEG as u64) << 4;
/// Offset of the `int 1Ah` handler within the shim.
const BIOS_SHIM_INT1A: u16 = 0x10;
/// BIOS ROM date string, read by illumos to decide whether ACPI can be used.
const BIOS_DATE_GPA: u64 = 0xFFFF5;
const BIOS_DATE: &[u8; 8] = b"09/27/26";
/// PC/AT model byte.
const BIOS_MODEL_GPA: u64 = 0xFFFFE;
const BIOS_MODEL_AT: u8 = 0xFC;
/// Where the BSP starts: the firmware prologue.
pub const MB2_PROLOGUE_GPA: u32 = 0xF8040;
/// Offset of the kernel entry immediate (`mov ecx, imm32`) in the prologue.
const PROLOGUE_ENTRY_OFF: usize = 0x3F;

/// 32-bit prologue, assembled from:
///
/// ```text
///     mov ebp, eax            ; keep the Multiboot2 magic
///     xor ecx, ecx            ; device 0..31 on bus 0, function 0
/// 1:  mov eax, ecx
///     shl eax, 11
///     or  eax, 0x80000000
///     mov esi, eax
///     mov dx, 0xcf8
///     out dx, eax
///     mov dx, 0xcfc
///     in  eax, dx
///     cmp ax, 0xffff          ; no device
///     je  2f
///     mov eax, esi
///     or  eax, 4              ; command register
///     mov dx, 0xcf8
///     out dx, eax
///     mov dx, 0xcfc
///     in  ax, dx
///     or  ax, 7               ; I/O, memory, bus master
///     out dx, ax
/// 2:  inc ecx
///     cmp ecx, 32
///     jb  1b
///     mov eax, ebp
///     mov ecx, <kernel entry>
///     jmp ecx
/// ```
const PROLOGUE: [u8; 0x45] = [
    0x89, 0xC5, 0x31, 0xC9, 0x89, 0xC8, 0xC1, 0xE0, 0x0B, 0x0D, 0x00, 0x00,
    0x00, 0x80, 0x89, 0xC6, 0x66, 0xBA, 0xF8, 0x0C, 0xEF, 0x66, 0xBA, 0xFC,
    0x0C, 0xED, 0x66, 0x83, 0xF8, 0xFF, 0x74, 0x16, 0x89, 0xF0, 0x83, 0xC8,
    0x04, 0x66, 0xBA, 0xF8, 0x0C, 0xEF, 0x66, 0xBA, 0xFC, 0x0C, 0x66, 0xED,
    0x66, 0x83, 0xC8, 0x07, 0x66, 0xEF, 0x41, 0x83, 0xF9, 0x20, 0x72, 0xC8,
    0x89, 0xE8, 0xB9, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xE1,
];

/// Real-mode handlers. Offset 0: any vector, `stc; mov ah,86h; retf 2`.
/// Offset 0x10: `int 1Ah`, PCI BIOS present for `AX=B101h` (AL=1:
/// mechanism 1, BX=0210h, CL=0: last bus, EDX='PCI '), otherwise
/// `stc; mov ah,81h; retf 2`. `retf 2` drops the pushed FLAGS so the
/// carry flag reaches the caller.
const BIOS_SHIM: [u8; 0x10 + 29] = [
    0xF9, 0xB4, 0x86, 0xCA, 0x02, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x3D,
    0x01, 0xB1, // cmp ax, 0xB101
    0x75, 0x12, // jne fail
    0xB8, 0x01, 0x00, // mov ax, 0x0001
    0xBB, 0x10, 0x02, // mov bx, 0x0210
    0xB1, 0x00, // mov cl, 0
    0x66, 0xBA, 0x50, 0x43, 0x49, 0x20, // mov edx, 'PCI '
    0xF8, // clc
    0xCA, 0x02, 0x00, // retf 2
    0xF9, // fail: stc
    0xB4, 0x81, // mov ah, 0x81
    0xCA, 0x02, 0x00, // retf 2
];

// ── Multiboot2 ABI ──────────────────────────────────────────────────

/// Header magic, searched for in the first 32 KiB of the image.
pub const MB2_HEADER_MAGIC: u32 = 0xE852_50D6;
/// Value in `EAX` on entry.
pub const MB2_BOOTLOADER_MAGIC: u32 = 0x36D7_6289;
const MB2_SEARCH: usize = 32768;
const MB2_ARCH_I386: u32 = 0;

const HDR_TAG_END: u16 = 0;
const HDR_TAG_ADDRESS: u16 = 2;
const HDR_TAG_ENTRY_ADDRESS: u16 = 3;

const TAG_END: u32 = 0;
const TAG_CMDLINE: u32 = 1;
const TAG_BOOT_LOADER_NAME: u32 = 2;
const TAG_MODULE: u32 = 3;
const TAG_BASIC_MEMINFO: u32 = 4;
const TAG_MMAP: u32 = 6;
const TAG_ACPI_OLD: u32 = 14;
const TAG_ACPI_NEW: u32 = 15;

const MMAP_ENTRY_SIZE: u32 = 24;
const ROOTFS_MODULE_CMDLINE: &str =
    "/platform/i86pc/amd64/boot_archive type=rootfs";
const LOADER_NAME: &str = "rshyve multiboot2";

// ── GDT ─────────────────────────────────────────────────────────────

/// 32-bit code: base 0, limit 0xFFFFF, type 0x0B, S=1, P=1, D/B=1, G=1.
const GDT_CODE32: u64 = 0x00CF_9B00_0000_FFFF;
const MB2_GDT_SIZE: usize = 24;
const MB2_GDTR_LIMIT: u32 = (MB2_GDT_SIZE as u32) - 1;

// ── Image ───────────────────────────────────────────────────────────

/// A kernel image carrying a Multiboot2 header with address and entry
/// address tags.
pub struct Multiboot2Kernel {
    data: ImageBytes,
    /// File offset of the first byte copied to `load_addr`.
    src_off: usize,
    load_addr: u32,
    /// Bytes copied from the file.
    load_len: usize,
    /// End of the zeroed BSS, or `load_addr + load_len` when there is none.
    bss_end: u32,
    entry: u32,
}

impl std::fmt::Debug for Multiboot2Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Multiboot2Kernel")
            .field("src_off", &self.src_off)
            .field("load_addr", &format_args!("{:#x}", self.load_addr))
            .field("load_len", &self.load_len)
            .field("bss_end", &format_args!("{:#x}", self.bss_end))
            .field("entry", &format_args!("{:#x}", self.entry))
            .finish()
    }
}

fn le32(data: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(off..off + 4)?.try_into().ok()?))
}

fn le16(data: &[u8], off: usize) -> Option<u16> {
    Some(u16::from_le_bytes(data.get(off..off + 2)?.try_into().ok()?))
}

/// File offset of a valid Multiboot2 header, if the image has one.
pub fn find_multiboot2_header(data: &[u8]) -> Option<usize> {
    let limit = data.len().min(MB2_SEARCH);
    (0..limit.saturating_sub(16)).step_by(8).find(|&off| {
        let (Some(magic), Some(arch), Some(len), Some(sum)) = (
            le32(data, off),
            le32(data, off + 4),
            le32(data, off + 8),
            le32(data, off + 12),
        ) else {
            return false;
        };
        magic == MB2_HEADER_MAGIC
            && magic.wrapping_add(arch).wrapping_add(len).wrapping_add(sum) == 0
    })
}

impl Multiboot2Kernel {
    pub fn from_bytes(data: impl Into<ImageBytes>) -> Result<Self> {
        let data = data.into();
        let bytes: &[u8] = &data;
        let hdr = find_multiboot2_header(bytes)
            .context("no Multiboot2 header in the first 32 KiB")?;
        let arch = le32(bytes, hdr + 4).expect("header was bounds-checked");
        ensure!(
            arch == MB2_ARCH_I386,
            "Multiboot2 architecture {arch} is not i386"
        );
        let hdr_len =
            le32(bytes, hdr + 8).expect("header was bounds-checked") as usize;
        let hdr_end = hdr
            .checked_add(hdr_len)
            .context("header length overflows")?;
        ensure!(
            hdr_end <= bytes.len(),
            "Multiboot2 header runs past the image"
        );

        let mut address = None;
        let mut entry = None;
        let mut tag = hdr + 16;
        while tag + 8 <= hdr_end {
            let ty = le16(bytes, tag).context("truncated header tag")?;
            let size =
                le32(bytes, tag + 4).context("truncated header tag")? as usize;
            ensure!(
                size >= 8 && tag + size <= hdr_end,
                "malformed header tag at {tag:#x}"
            );
            match ty {
                HDR_TAG_END => break,
                HDR_TAG_ADDRESS => {
                    ensure!(size >= 24, "short address tag");
                    address = Some([
                        le32(bytes, tag + 8).expect("bounded"),
                        le32(bytes, tag + 12).expect("bounded"),
                        le32(bytes, tag + 16).expect("bounded"),
                        le32(bytes, tag + 20).expect("bounded"),
                    ]);
                }
                HDR_TAG_ENTRY_ADDRESS => {
                    ensure!(size >= 12, "short entry address tag");
                    entry = Some(le32(bytes, tag + 8).expect("bounded"));
                }
                _ => {}
            }
            tag += (size + 7) & !7;
        }

        // ELF-only Multiboot2 kernels need a loader that interprets the
        // program headers; the tags make placement explicit and are what
        // illumos provides, so require them.
        let [header_addr, load_addr, load_end_addr, bss_end_addr] =
            address.context("Multiboot2 header has no address tag")?;
        let entry =
            entry.context("Multiboot2 header has no entry address tag")?;
        let delta = header_addr
            .checked_sub(load_addr)
            .context("header_addr below load_addr")?
            as usize;
        let src_off = hdr
            .checked_sub(delta)
            .context("load_addr precedes the start of the image")?;
        let load_len = if load_end_addr == 0 {
            bytes.len() - src_off
        } else {
            load_end_addr
                .checked_sub(load_addr)
                .context("load_end_addr below load_addr")? as usize
        };
        ensure!(
            src_off + load_len <= bytes.len(),
            "load range exceeds the image"
        );
        let load_end = load_addr as u64 + load_len as u64;
        ensure!(
            load_end <= u64::from(u32::MAX),
            "image does not fit below 4 GiB"
        );
        let bss_end = if bss_end_addr == 0 {
            load_end as u32
        } else {
            bss_end_addr
        };
        ensure!(
            u64::from(bss_end) >= load_end,
            "bss_end_addr below the loaded image"
        );
        ensure!(
            entry >= load_addr && u64::from(entry) < load_end,
            "entry {entry:#x} lies outside the loaded image"
        );
        Ok(Self {
            data,
            src_off,
            load_addr,
            load_len,
            bss_end,
            entry,
        })
    }

    pub fn entry_point_32(&self) -> u32 {
        self.entry
    }

    /// First byte past the image and its BSS.
    pub fn load_end(&self) -> u64 {
        u64::from(self.bss_end)
    }
}

// ── MBI builder ─────────────────────────────────────────────────────

struct Mbi(Vec<u8>);

impl Mbi {
    fn new() -> Self {
        // total_size and reserved, filled in by finish().
        Self(vec![0; 8])
    }

    fn tag(&mut self, ty: u32, body: &[u8]) {
        let size = 8 + body.len();
        self.0.extend_from_slice(&ty.to_le_bytes());
        self.0.extend_from_slice(&(size as u32).to_le_bytes());
        self.0.extend_from_slice(body);
        while !self.0.len().is_multiple_of(8) {
            self.0.push(0);
        }
    }

    fn string_tag(&mut self, ty: u32, s: &str) {
        let mut body = s.as_bytes().to_vec();
        body.push(0);
        self.tag(ty, &body);
    }

    fn finish(mut self) -> Vec<u8> {
        self.tag(TAG_END, &[]);
        let total = self.0.len() as u32;
        self.0[0..4].copy_from_slice(&total.to_le_bytes());
        self.0
    }
}

/// Page-aligned module address directly above `load_end`, below the
/// lowmem ceiling.
fn place_module(mem_size: usize, len: usize, load_end: u64) -> Result<u64> {
    let addr = load_end
        .checked_add(0xFFF)
        .context("module placement overflows")?
        & !0xFFF;
    let end = addr
        .checked_add(len as u64)
        .context("module placement overflows")?;
    let lowmem_limit = (mem_size as u64).min(crate::LOWMEM_LIMIT);
    ensure!(
        end <= lowmem_limit,
        "module of {len} bytes does not fit above the kernel (need more RAM)"
    );
    Ok(addr)
}

/// Load a Multiboot2 kernel, an optional rootfs module, the GDT and MBI.
///
/// Returns the 32-bit entry point. Must run after the ACPI tables are
/// placed: the RSDP is copied into the MBI.
pub fn load_multiboot2_kernel(
    mem: &MemCtx,
    kernel: &Multiboot2Kernel,
    cmdline: &str,
    initrd: Option<&InitrdImage>,
    mem_size: usize,
    acpi_rsdp_addr: u64,
) -> Result<u32> {
    // 1. The image, straight from the file when there is one. The BSS
    //    relies on guest RAM starting zeroed.
    let load_addr = u64::from(kernel.load_addr);
    match kernel.data.file() {
        Some(file) => mem
            .load_from_file(
                load_addr,
                file,
                kernel.src_off as u64,
                kernel.load_len,
            )
            .context("failed to load the Multiboot2 image")?,
        None => {
            let bytes =
                &kernel.data[kernel.src_off..kernel.src_off + kernel.load_len];
            mem.write_bulk(load_addr, bytes)
                .context("failed to write the Multiboot2 image")?;
        }
    }

    ensure!(
        cmdline.len() < CMDLINE_MAX,
        "cmdline too long (max {CMDLINE_MAX})"
    );

    // 2. The module, page-aligned directly above the kernel, as the
    //    illumos loader places it. dboot allocates its own memory upward
    //    from the end of the last module, so a module at the top of RAM
    //    (the Linux initrd placement) leaves dboot nothing to allocate.
    let module = match initrd {
        Some(rd) => {
            let addr = place_module(mem_size, rd.len(), kernel.load_end())?;
            mem.write_bulk(addr, rd.as_bytes())
                .context("failed to write the module")?;
            Some((addr, rd.len() as u64))
        }
        None => None,
    };

    // 3. Legacy BIOS shim and the IVT that points into it.
    mem.write(BIOS_SHIM_GPA, &BIOS_SHIM)
        .context("failed to write the BIOS shim")?;
    let mut ivt = [0u8; 256 * 4];
    for (vector, entry) in ivt.chunks_exact_mut(4).enumerate() {
        let offset = if vector == 0x1A { BIOS_SHIM_INT1A } else { 0 };
        entry[0..2].copy_from_slice(&offset.to_le_bytes());
        entry[2..4].copy_from_slice(&BIOS_SHIM_SEG.to_le_bytes());
    }
    mem.write(0, &ivt).context("failed to write the IVT")?;
    mem.write(BIOS_DATE_GPA, BIOS_DATE)
        .context("failed to write the BIOS date")?;
    mem.write(BIOS_MODEL_GPA, &[BIOS_MODEL_AT])
        .context("failed to write the BIOS model byte")?;
    let mut prologue = PROLOGUE;
    prologue[PROLOGUE_ENTRY_OFF..PROLOGUE_ENTRY_OFF + 4]
        .copy_from_slice(&kernel.entry_point_32().to_le_bytes());
    mem.write(u64::from(MB2_PROLOGUE_GPA), &prologue)
        .context("failed to write the firmware prologue")?;

    // 4. GDT.
    let mut gdt = [0u8; MB2_GDT_SIZE];
    gdt[8..16].copy_from_slice(&GDT_CODE32.to_le_bytes());
    gdt[16..24].copy_from_slice(&GDT_DATA.to_le_bytes());
    mem.write(MB2_GPA_GDT, &gdt)
        .context("failed to write GDT")?;

    // 5. MBI.
    let e820 = acpi::build_e820_entries(mem_size);
    let mut mbi = Mbi::new();
    mbi.string_tag(TAG_CMDLINE, cmdline);
    mbi.string_tag(TAG_BOOT_LOADER_NAME, LOADER_NAME);
    if let Some((addr, len)) = module {
        let end = addr + len;
        ensure!(
            end <= u64::from(u32::MAX),
            "module does not fit below 4 GiB"
        );
        let mut body = Vec::new();
        body.extend_from_slice(&(addr as u32).to_le_bytes());
        body.extend_from_slice(&(end as u32).to_le_bytes());
        body.extend_from_slice(ROOTFS_MODULE_CMDLINE.as_bytes());
        body.push(0);
        mbi.tag(TAG_MODULE, &body);
    }

    // Basic meminfo: KiB below 1 MiB and KiB of contiguous RAM above it.
    let mut lower = 0u64;
    let mut upper = 0u64;
    for entry in &e820 {
        let (addr, size, type_) = (entry.addr, entry.size, entry.type_);
        if type_ != 1 {
            continue;
        }
        if addr == 0 {
            lower = size / 1024;
        } else if addr == 0x10_0000 {
            upper = size / 1024;
        }
    }
    let mut meminfo = Vec::new();
    meminfo.extend_from_slice(&(lower.min(640) as u32).to_le_bytes());
    meminfo.extend_from_slice(
        &(upper.min(u64::from(u32::MAX)) as u32).to_le_bytes(),
    );
    mbi.tag(TAG_BASIC_MEMINFO, &meminfo);

    let mut mmap = Vec::new();
    mmap.extend_from_slice(&MMAP_ENTRY_SIZE.to_le_bytes());
    mmap.extend_from_slice(&0u32.to_le_bytes()); // entry_version
    for entry in &e820 {
        // E820Entry is packed, so copy each field out before use. The
        // Multiboot2 memory types are the E820 types.
        let (addr, size, type_) = (entry.addr, entry.size, entry.type_);
        mmap.extend_from_slice(&addr.to_le_bytes());
        mmap.extend_from_slice(&size.to_le_bytes());
        mmap.extend_from_slice(&type_.to_le_bytes());
        mmap.extend_from_slice(&0u32.to_le_bytes());
    }
    mbi.tag(TAG_MMAP, &mmap);

    if acpi_rsdp_addr != 0 {
        let mut rsdp = [0u8; 36];
        mem.read(acpi_rsdp_addr, &mut rsdp)
            .context("failed to read the RSDP")?;
        match rsdp[15] {
            0 => mbi.tag(TAG_ACPI_OLD, &rsdp[..20]),
            rev if rev >= 2 => mbi.tag(TAG_ACPI_NEW, &rsdp),
            rev => bail!("unexpected RSDP revision {rev}"),
        }
    }

    let mbi = mbi.finish();
    ensure!(
        mbi.len() <= MB2_MBI_MAX,
        "MBI of {} bytes overruns the GDT",
        mbi.len()
    );
    mem.write(MB2_GPA_MBI, &mbi)
        .context("failed to write the MBI")?;

    Ok(kernel.entry_point_32())
}

// ── BSP setup ───────────────────────────────────────────────────────

/// Configure vCPU 0 for Multiboot2 entry: 32-bit protected mode, paging
/// off, `EAX` = bootloader magic, `EBX` = MBI.
pub fn setup_multiboot2_bsp<V: BootVcpu + ?Sized>(
    vcpu: &V,
    entry32: u32,
) -> Result<()> {
    use bhyve_api::{
        vm_reg_name::*, SEG_ACCESS_DB, SEG_ACCESS_G, SEG_ACCESS_P, SEG_ACCESS_S,
    };

    vcpu.reboot_state().context("reboot_state")?;

    vcpu.set_reg(VM_REG_GUEST_CR0, CR0_PE | CR0_ET)?;
    vcpu.set_reg(VM_REG_GUEST_CR3, 0)?;
    vcpu.set_reg(VM_REG_GUEST_CR4, 0)?;
    vcpu.set_reg(VM_REG_GUEST_EFER, 0)?;

    bsp::program_flat_segments(
        vcpu,
        &FlatSegments {
            code_access: SEG_ACCESS_P
                | SEG_ACCESS_S
                | SEG_ACCESS_DB
                | SEG_ACCESS_G
                | SEG_TYPE_CODE_XRA,
            gdt_base: MB2_GPA_GDT,
            gdt_limit: MB2_GDTR_LIMIT,
        },
    )?;

    vcpu.set_reg(VM_REG_GUEST_RIP, u64::from(entry32))?;
    vcpu.set_reg(VM_REG_GUEST_RFLAGS, RFLAGS_RESERVED_BIT1)?;
    vcpu.set_reg(VM_REG_GUEST_RBX, MB2_GPA_MBI)?;
    bsp::zero_gprs(vcpu, VM_REG_GUEST_RBX)?;
    vcpu.set_reg(VM_REG_GUEST_RAX, u64::from(MB2_BOOTLOADER_MAGIC))?;

    vcpu.set_run_state(bhyve_api::VRS_RUN)?;
    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic image laid out like illumos amd64 `unix`: the header at
    /// file offset 0x190 with header_addr 0xc00038, so load_addr maps file
    /// offset 0.
    fn image() -> Vec<u8> {
        let mut img = vec![0u8; 0x1000];
        let hdr = 0x190;
        let tags: Vec<u8> = [
            // address tag
            &[2u8, 0, 0, 0, 24, 0, 0, 0][..],
            &0xc0_0038u32.to_le_bytes(),
            &0xbf_fea8u32.to_le_bytes(),
            &0u32.to_le_bytes(),
            &0u32.to_le_bytes(),
            // entry address tag, padded to 8
            &[3u8, 0, 0, 0, 12, 0, 0, 0],
            &0xc0_0100u32.to_le_bytes(),
            &[0u8; 4],
            // end tag
            &[0u8, 0, 0, 0, 8, 0, 0, 0],
        ]
        .concat();
        let len = 16 + tags.len() as u32;
        let sum = 0u32.wrapping_sub(MB2_HEADER_MAGIC.wrapping_add(len));
        img[hdr..hdr + 4].copy_from_slice(&MB2_HEADER_MAGIC.to_le_bytes());
        img[hdr + 8..hdr + 12].copy_from_slice(&len.to_le_bytes());
        img[hdr + 12..hdr + 16].copy_from_slice(&sum.to_le_bytes());
        img[hdr + 16..hdr + 16 + tags.len()].copy_from_slice(&tags);
        img
    }

    #[test]
    fn finds_header_and_derives_placement() {
        let k = Multiboot2Kernel::from_bytes(image()).expect("parsed");
        assert_eq!(k.src_off, 0);
        assert_eq!(k.load_addr, 0xbf_fea8);
        assert_eq!(k.load_len, 0x1000);
        assert_eq!(k.entry_point_32(), 0xc0_0100);
        assert_eq!(k.load_end(), 0xbf_fea8 + 0x1000);
    }

    #[test]
    fn module_sits_directly_above_the_kernel() {
        let addr = place_module(64 << 20, 4096, 0xc0_1234).expect("placed");
        assert_eq!(addr, 0xc0_2000);
        let err = place_module(16 << 20, 8 << 20, 0xc0_0000)
            .expect_err("module past the end of RAM");
        assert!(err.to_string().contains("does not fit"), "{err}");
    }

    #[test]
    fn bios_shim_jump_lands_on_the_fail_path() {
        // jne at 0x13 with rel8 0x12 -> target 0x15 + 0x12 = 0x27.
        let jne = 0x13;
        assert_eq!(BIOS_SHIM[jne], 0x75);
        let target = jne + 2 + BIOS_SHIM[jne + 1] as usize;
        assert_eq!(&BIOS_SHIM[target..target + 3], &[0xF9, 0xB4, 0x81]);
        assert_eq!(BIOS_SHIM_GPA, 0xF8000);
    }

    #[test]
    fn prologue_matches_its_listing() {
        // mov ecx, imm32 then jmp ecx at the end; the je/jb targets stay
        // inside the loop.
        assert_eq!(PROLOGUE[PROLOGUE_ENTRY_OFF - 1], 0xB9);
        assert_eq!(&PROLOGUE[PROLOGUE.len() - 2..], &[0xFF, 0xE1]);
        let je = 0x1E;
        assert_eq!(PROLOGUE[je], 0x74);
        assert_eq!(je + 2 + PROLOGUE[je + 1] as usize, 0x36); // inc ecx
        let jb = 0x3A;
        assert_eq!(PROLOGUE[jb], 0x72);
        assert_eq!((jb as i32) + 2 + (PROLOGUE[jb + 1] as i8) as i32, 0x04);
        // The prologue sits after the BIOS shim.
        assert!(
            BIOS_SHIM_GPA as usize + BIOS_SHIM.len()
                <= MB2_PROLOGUE_GPA as usize
        );
    }

    #[test]
    fn rejects_bad_checksum() {
        let mut img = image();
        img[0x19c] ^= 1;
        assert!(find_multiboot2_header(&img).is_none());
    }

    #[test]
    fn rejects_entry_outside_image() {
        let mut img = image();
        // entry tag body starts at 0x190 + 16 + 24 + 8
        let at = 0x190 + 16 + 24 + 8;
        img[at..at + 4].copy_from_slice(&0x10_0000u32.to_le_bytes());
        let err = Multiboot2Kernel::from_bytes(img).expect_err("entry outside");
        assert!(
            err.to_string().contains("outside the loaded image"),
            "{err}"
        );
    }

    #[test]
    fn mbi_tags_are_eight_byte_aligned_and_sized() {
        let mut mbi = Mbi::new();
        mbi.string_tag(TAG_CMDLINE, "unix -B console=ttya");
        mbi.tag(TAG_BASIC_MEMINFO, &[0; 8]);
        let out = mbi.finish();
        assert_eq!(
            u32::from_le_bytes(out[0..4].try_into().unwrap()) as usize,
            out.len()
        );
        let mut off = 8;
        let mut types = Vec::new();
        while off < out.len() {
            let ty = u32::from_le_bytes(out[off..off + 4].try_into().unwrap());
            let size =
                u32::from_le_bytes(out[off + 4..off + 8].try_into().unwrap())
                    as usize;
            types.push(ty);
            off += (size + 7) & !7;
            assert_eq!(off % 8, 0);
        }
        assert_eq!(types, vec![TAG_CMDLINE, TAG_BASIC_MEMINFO, TAG_END]);
    }
}
