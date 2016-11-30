# TransitLCApprox
Limb darkened transit model under the small planet approximation from Mandel and Agol 2002.

For an example of using the routines run
python smalltransit.py
Handles nonlinear 4-parameter and quadratic 2-parameter limb darkening laws
function arguments are as defined in mandel and Agol 2002.
They are in terms of planet star separation normalized to star radius, z, and planet to star radius ratio, k
Additional function available midpoint_transit_depth()
that provides depth for the exactly b=0 case.
