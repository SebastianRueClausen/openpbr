use glam::Vec3;

// Fraunhofer lines mapped to R/G/B channels.
const WAVELENGTH_R: f32 = 656.3; // C line (red)
const WAVELENGTH_G: f32 = 587.6; // d line (green, the Cauchy reference wavelength)
const WAVELENGTH_B: f32 = 486.1; // F line (blue)

/// Per-channel (RGB) IORs using Cauchy's dispersion model.
///
/// `dispersion` is the OpenPBR dispersion parameter, where 0 means no
/// dispersion and 1 corresponds to the highest realistic dispersion.
pub fn rgb_iors(ior: f32, dispersion: f32) -> Vec3 {
    if dispersion == 0.0 {
        return Vec3::splat(ior);
    }
    Vec3::new(
        cauchy_ior(ior, dispersion, WAVELENGTH_R),
        cauchy_ior(ior, dispersion, WAVELENGTH_G),
        cauchy_ior(ior, dispersion, WAVELENGTH_B),
    )
}

/// Computes a single wavelength-adjusted IOR via Cauchy's equation.
///
/// Handles `ior < 1` by reflecting across 1 (flipping to above-one, computing,
/// then inverting), matching Adobe.
fn cauchy_ior(ior: f32, dispersion: f32, wavelength_nm: f32) -> f32 {
    let flipped = ior < 1.0;
    let n_d = if flipped { 1.0 / ior } else { ior };

    // Abbe number from dispersion parameter (V_d = 20 / dispersion).
    let v_d = 20.0 / dispersion;

    // Cauchy coefficients A and B chosen so that n(d) = n_d and
    // the Abbe number equals V_d exactly.
    let b = (n_d - 1.0) / (v_d * (1.0 / WAVELENGTH_B.powi(2) - 1.0 / WAVELENGTH_R.powi(2)));
    let a = n_d - b / WAVELENGTH_G.powi(2);

    let n = a + b / wavelength_nm.powi(2);
    if flipped {
        1.0 / n
    } else {
        n
    }
}
