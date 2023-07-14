use std::env;

fn main() {
    enable_simd();
    enable_libc();
}

// This adds various simd cfgs if this compiler and target support it.
//
// This can be disabled with RUSTFLAGS="--cfg memchr_disable_auto_simd", but
// this is generally only intended for testing.
//
// On targets which don't feature SSE2, this is disabled, as LLVM wouln't know
// how to work with SSE2 operands. Enabling SSE4.2 and AVX on SSE2-only targets
// is not a problem. In that case, the fastest option will be chosen at
// runtime.
fn enable_simd() {
    // miri is not yet equipped to deal with SIMD stuff, so we just shut all
    // of it off. It would be great to have support though, because it is
    // precisely the code we really want to check for UB due to the sprawling
    // use of not-safe vector routines.
    if is_miri() {
        return;
    }
    // If automatic SIMD was disabled then we respect that, but this is only
    // intended for testing. Basically, setting this lets us set any of the cfg
    // knobs below that we want to carve a path through the code that lets us
    // test things. Otherwise, for example, it would be impossible to test the
    // SSE2 routine on a target with AVX2 support. (An alternative would be
    // to expose something in the API of the crate to tweak this, but I don't
    // think that's worth doing as it would likely greatly complicate things.)
    if is_env_set("CARGO_CFG_MEMCHR_DISABLE_AUTO_SIMD") {
        return;
    }
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    match &arch[..] {
        "x86_64" => {
            if is_feature_set("STD") {
                println!("cargo:rustc-cfg=memchr_Rsse42");
                println!("cargo:rustc-cfg=memchr_Ravx2");
            }
            if target_has_feature("sse2") {
                println!("cargo:rustc-cfg=memchr_Csse2");
            }
            if target_has_feature("sse4.2") {
                println!("cargo:rustc-cfg=memchr_Csse42");
            }
            if target_has_feature("avx2") {
                println!("cargo:rustc-cfg=memchr_Cavx2");
            }
            if !target_has_feature("sse2") {
                return;
            }
            println!("cargo:rustc-cfg=memchr_runtime_simd");
            println!("cargo:rustc-cfg=memchr_runtime_sse2");
            println!("cargo:rustc-cfg=memchr_runtime_sse42");
            println!("cargo:rustc-cfg=memchr_runtime_avx");
        }
        "wasm32" | "wasm64" => {
            if target_has_feature("simd128") {
                println!("cargo:rustc-cfg=memchr_Csimd128");
            }
            if !target_has_feature("simd128") {
                return;
            }
            println!("cargo:rustc-cfg=memchr_runtime_simd");
            println!("cargo:rustc-cfg=memchr_runtime_wasm128");
        }
        "aarch64" => {
            // We can technically query for neon at runtime, but neon is a
            // required part of aarch64. So no such querying is seemingly
            // required. We just need to make sure neon is in the supported
            // target features at compile time. (It isn't clear when neon
            // wouldn't be in the supported target features for aarch64,
            // perhaps never, but it's queryable so we query it and respect the
            // result.)
            if target_has_feature("neon") {
                println!("cargo:rustc-cfg=memchr_Cneon");
            }
        }
        _ => {}
    }
}

// This adds a `memchr_libc` cfg if and only if libc can be used, if no other
// better option is available.
//
// This could be performed in the source code, but it's simpler to do it once
// here and consolidate it into one cfg knob.
//
// Basically, we use libc only if its enabled and if we aren't targeting a
// known bad platform. For example, wasm32 doesn't have a libc and the
// performance of memchr on Windows is seemingly worse than the fallback
// implementation.
fn enable_libc() {
    const NO_ARCH: &'static [&'static str] = &["wasm32", "windows"];
    const NO_ENV: &'static [&'static str] = &["sgx"];

    if !is_feature_set("LIBC") {
        return;
    }

    let arch = match env::var("CARGO_CFG_TARGET_ARCH") {
        Err(_) => return,
        Ok(arch) => arch,
    };
    let env = match env::var("CARGO_CFG_TARGET_ENV") {
        Err(_) => return,
        Ok(env) => env,
    };
    if NO_ARCH.contains(&&*arch) || NO_ENV.contains(&&*env) {
        return;
    }

    println!("cargo:rustc-cfg=memchr_libc");
}

fn is_feature_set(name: &str) -> bool {
    is_env_set(&format!("CARGO_FEATURE_{}", name))
}

fn is_env_set(name: &str) -> bool {
    env::var_os(name).is_some()
}

fn target_has_feature(feature: &str) -> bool {
    env::var("CARGO_CFG_TARGET_FEATURE")
        .map(|features| features.contains(feature))
        .unwrap_or(false)
}

fn is_miri() -> bool {
    env::var_os("CARGO_CFG_MIRI").is_some()
}
