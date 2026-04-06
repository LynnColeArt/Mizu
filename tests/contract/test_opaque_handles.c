#include "mizu.h"

/*
 * This file exists to anchor the opaque-handle contract in a real translation
 * unit. Consumers should only be able to name pointers to these handles, not
 * instantiate them directly.
 */

int main(void) {
    mizu_runtime_t *runtime = 0;
    mizu_model_t *model = 0;
    mizu_session_t *session = 0;

    return (runtime == 0 && model == 0 && session == 0) ? 0 : 1;
}
