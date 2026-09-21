// See eclipse/catalog.hpp. Phase 4 work package A ships the contract and
// compiling stubs; work package B ports app/catalog.py into them, keeping the
// operation-order rules listed in the header.
#include "eclipse/catalog.hpp"

#include <stdexcept>

namespace eclipse::catalog {

namespace {
[[noreturn]] void not_ported(const char* what) {
    throw std::logic_error(std::string("eclipse::catalog::") + what + ": phase 4 WP-B");
}
}  // namespace

std::vector<double> rho_at(std::span<const double>) { not_ported("rho_at"); }

Sources sources_from_ephem(Frame) { not_ported("sources_from_ephem"); }

std::vector<double> coarse_grid(double, double) { not_ported("coarse_grid"); }

std::vector<std::size_t> local_minima(std::span<const double>) { not_ported("local_minima"); }

std::vector<double> scan_candidates(double, double, const RhoAt&) { not_ported("scan_candidates"); }

std::vector<double> refine_greatest(std::span<const double>, const RhoAt&) {
    not_ported("refine_greatest");
}

Classification classify(double, double, double, double, double, bool, double) {
    not_ported("classify");
}

bool hybrid(double, double, const ElementsAtEt&) { not_ported("hybrid"); }

Detail add_detail(double, Frame, bool, double, double) { not_ported("add_detail"); }

std::vector<Event> find_eclipses(double, double, const Sources&, bool, int) {
    not_ported("find_eclipses");
}

std::vector<Event> find_eclipses(double, double, Frame, bool, int) { not_ported("find_eclipses"); }

}  // namespace eclipse::catalog
