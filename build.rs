// SPDX-License-Identifier: Apache-2.0

fn main() {
    #[cfg(feature = "proto")]
    {
        println!("cargo:rerun-if-changed=proto/match_recognize.proto");
        prost_build::compile_protos(&["proto/match_recognize.proto"], &["proto"])
            .expect("failed to compile match_recognize.proto");
    }
}
