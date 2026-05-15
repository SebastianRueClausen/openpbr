fn main() {
    let glm_include = find_glm().unwrap_or_else(|| {
        panic!(
            "GLM not found. Install with:\n  macOS:  brew install glm\n  Ubuntu: apt install libglm-dev\nOr set the GLM_INCLUDE_DIR environment variable."
        )
    });

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(".")
        .include("src")
        .include(&glm_include)
        .warnings(false)
        .file("src/bridge.cpp")
        .compile("openpbr_bridge");

    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed=src/bridge.h");
    println!("cargo:rerun-if-changed=openpbr-bsdf");
}

fn find_glm() -> Option<String> {
    if let Ok(dir) = std::env::var("GLM_INCLUDE_DIR") {
        if std::path::Path::new(&format!("{dir}/glm/glm.hpp")).exists() {
            return Some(dir);
        }
    }

    if let Ok(prefix) = std::env::var("HOMEBREW_PREFIX") {
        let dir = format!("{prefix}/include");
        if std::path::Path::new(&format!("{dir}/glm/glm.hpp")).exists() {
            return Some(dir);
        }
    }

    for candidate in [
        "/opt/homebrew/include",
        "/usr/local/include",
        "/usr/include",
    ] {
        if std::path::Path::new(&format!("{candidate}/glm/glm.hpp")).exists() {
            return Some(candidate.to_owned());
        }
    }

    None
}
