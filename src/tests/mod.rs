pub(crate) mod memchr;
mod memchr2;
pub(crate) mod packedpair;
#[macro_use]
pub(crate) mod substring;

// For debugging, particularly in CI, print out the byte order of the current
// target.
#[cfg(all(feature = "std", target_endian = "little"))]
#[test]
fn byte_order() {
    eprintln!("LITTLE ENDIAN");
}

#[cfg(all(feature = "std", target_endian = "big"))]
#[test]
fn byte_order() {
    eprintln!("BIG ENDIAN");
}
