// See eclipse/circumstances.hpp. Phase 3, work package A: the contract
// compiles; every body is a stub that work package B replaces with the port
// of app/circumstances.py, in the oracle's exact operation order (roadmap §7).
// No SPICE here: the ephemeris enters only through ``Model``'s callables,
// which ``model_from_ephem`` binds to ``ephem``.
#include "eclipse/circumstances.hpp"

#include <stdexcept>

namespace eclipse::circumstances {

namespace {
[[noreturn]] void not_yet() { throw std::logic_error("phase 3 WP-B"); }
}  // namespace

Model model_from_ephem(double, Frame, double) { not_yet(); }

Series series(const Elements&, double, double) { not_yet(); }

Series series(const Elements&, std::span<const double>, std::span<const double>) { not_yet(); }

double overlap_area(double, double, double) { not_yet(); }

double obscuration(double, double, double) { not_yet(); }

AltAz sun_altaz(double, double, double, double) { not_yet(); }

double sun_alt_grid(double, double, double, double) { not_yet(); }

std::vector<Root> roots(std::span<const double>, std::span<const double>) { not_yet(); }

Bracketed bracketed_series(const Model&, double, double) { not_yet(); }

std::vector<double> refine_contacts(const Model&, double, double, std::span<const double>,
                                    std::span<const double>, std::span<const std::uint8_t>) {
    not_yet();
}

double refine_maximum(std::span<const double>, std::span<const double>, std::size_t) { not_yet(); }

LocalRaw local_circumstances(const Model&, double, double) { not_yet(); }

LocalRaw local_circumstances(double, Frame, double, double, double) { not_yet(); }

GridResult circumstances_grid(const Model&, std::span<const double>, std::span<const double>, double,
                              int) {
    not_yet();
}

GridResult circumstances_grid(double, Frame, double, std::span<const double>,
                              std::span<const double>, double, int) {
    not_yet();
}

}  // namespace eclipse::circumstances
