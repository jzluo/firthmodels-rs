fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(accelerate)");
    // Every Apple silicon Mac has Accelerate. Other targets use faer and the
    // fearless_simd kernels.
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    if os == "macos" && arch == "aarch64" {
        println!("cargo::rustc-cfg=accelerate");
    }
}
