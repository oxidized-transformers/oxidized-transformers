use std::path::PathBuf;

use snafu::{report, ResultExt, Whatever};

static KERNELS: &[&str] = &["src/nonzero.cu"];

#[report]
fn main() -> Result<(), Whatever> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/step_output_iterator.cuh");
    for kernel in KERNELS {
        println!("cargo:rerun-if-changed={kernel}");
    }

    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(KERNELS.iter().collect())
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--verbose");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").whatever_context("OUT_DIR not set")?);
    builder.build_lib(out_dir.join("liboxidized_cuda_kernels.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=oxidized_cuda_kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}
