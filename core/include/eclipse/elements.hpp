// Besselian elements as struct-of-arrays (roadmap §3) — the dict returned by
// app/ephemeris.besselian_instants, one vector per key.
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace eclipse {

/// Earth-orientation frame (app/ephemeris.EARTH_FRAMES).
enum class Frame {
    ITRS,       ///< ERFA c2t06a (IAU 2006/2000A) + IERS EOP [SOFA][IERS2010]; default
    TOD,        ///< true equator & equinox of date: pnm06a + gst06a, no polar motion
    ITRF93,     ///< SPICE body-fixed frame from the binary Earth PCK
    IAU_EARTH,  ///< SPICE low-precision analytic rotation
};

inline Frame frame_from_string(std::string_view s) {
    if (s == "ITRS") return Frame::ITRS;
    if (s == "TOD") return Frame::TOD;
    if (s == "ITRF93") return Frame::ITRF93;
    if (s == "IAU_EARTH") return Frame::IAU_EARTH;
    throw std::invalid_argument("unknown earth_frame '" + std::string(s) +
                                "' (ITRS, TOD, ITRF93, IAU_EARTH)");
}

inline const char* to_string(Frame f) {
    switch (f) {
    case Frame::ITRS: return "ITRS";
    case Frame::TOD: return "TOD";
    case Frame::ITRF93: return "ITRF93";
    case Frame::IAU_EARTH: return "IAU_EARTH";
    }
    return "?";
}

/// Besselian elements at n instants. Units as in app/ephemeris.BesselianInstant:
/// ``x, y, z, l1, l2`` in Earth equatorial radii, ``d, mu`` in **degrees**
/// (``mu`` wrapped to (-180, 180]), ``tan_f1, tan_f2`` dimensionless.
struct Elements {
    std::vector<double> x, y, z, d, mu, l1, l2, tan_f1, tan_f2;

    size_t size() const noexcept { return x.size(); }
    void reserve(size_t n);
};

inline void Elements::reserve(size_t n) {
    for (auto* v : {&x, &y, &z, &d, &mu, &l1, &l2, &tan_f1, &tan_f2}) v->reserve(n);
}

}  // namespace eclipse
