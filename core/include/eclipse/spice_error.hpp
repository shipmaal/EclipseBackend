// SPICE error surface of libeclipse.
//
// CSPICE's default error action aborts the process. libeclipse switches the
// toolkit to RETURN mode once (like spiceypy does at import) and converts every
// failed call into this exception, so a missing kernel is a catchable error in
// C++ and in Python, never a crash. See docs/CPP_ROADMAP.md §2.
#pragma once

#include <stdexcept>
#include <string>

namespace eclipse {

/// A CSPICE call signalled an error (``failed_c()`` after the call).
///
/// Fields mirror what spiceypy's ``check_for_spice_error`` collects
/// (``getmsg_c`` SHORT / EXPLAIN / LONG and ``qcktrc_c``); ``what()`` is the
/// same multi-line text spiceypy prints for a ``SpiceyError``.
class spice_error : public std::runtime_error {
public:
    spice_error(std::string short_msg, std::string explain, std::string long_msg,
                std::string traceback);

    const std::string& short_message() const noexcept { return short_; }
    const std::string& explanation() const noexcept { return explain_; }
    const std::string& long_message() const noexcept { return long_; }
    const std::string& traceback() const noexcept { return traceback_; }

private:
    std::string short_, explain_, long_, traceback_;
};

}  // namespace eclipse
