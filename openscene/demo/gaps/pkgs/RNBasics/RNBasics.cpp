/* Source file for GAPS basics module */



/* Include files */

#include "RNBasics.h"



// Namespace

namespace gaps {



/* Private variables */

static int RNbasics_active_count = 0;



int RNInitBasics(void)
{
    // Check whether are already initialized 
    if ((RNbasics_active_count++) > 0) return TRUE;

    // Initialize submodules
    // ???
    
    // Seed random number generator
    RNSeedRandomScalar();

    // Do not call RNInitGrfx here
    // (it needs to be called after rendering context is created)
    
    // Return OK status 
    return TRUE;
}



void RNStopBasics(void)
{
    // Check whether have been initialized 
    if ((--RNbasics_active_count) > 0) return;

    // Stop submodules
    // ???
}



} // namespace gaps
