#include "ElpMpp02.h"
#include <stdlib.h>

int main(int argc, char *argv[]) {
   if (argc != 2) {
      printf("Usage: %s <JD>\n", argv[0]);
      exit(1);
   }
   Elp_facs facs; 
   Elp_paras paras;
   Elp_coefs coefs;

   int corr;
   double X,Y,Z, JD, T;

// Convert the command-line argument to a double
   JD = atof(argv[1]);

//-----------------------------------------------------------------------
//     Compute Lunar coordinates (J2000.0 mean ecliptic and equinox):
//     Parameters fitted to LLR observations.
//-----------------------------------------------------------------------
   corr = 0; // use parameters fitted to the LLR data
   setup_parameters(corr, paras, facs);
   setup_Elp_coefs(coefs, facs);
   T = (JD - 2451545.0)/36525.0;
   getX2000(T, paras, coefs, X,Y,Z);
   printf("-------------------------------------------------------\n");
   printf("Lunar coordinates (J2000.0 mean ecliptic and equinox):\n");
   printf("Parameters fitted to LLR observations.\n");
   printf("-------------------------------------------------------\n");
   printf("JD: %f\n", JD);
   printf("X, Y, Z (km): %f   %f   %f\n\n", X,Y,Z);

//-----------------------------------------------------------------------
//     Compute Lunar coordinates (J2000.0 mean ecliptic and equinox):
//     Parameters fitted to JPL ephemeris DE405/DE406.
//-----------------------------------------------------------------------
   corr = 1; // use parameters fitted to DE405
   setup_parameters(corr, paras, facs);
   setup_Elp_coefs(coefs, facs);
   T = (JD - 2451545.0)/36525.0;
   getX2000(T, paras, coefs, X,Y,Z);
   printf("-------------------------------------------------------\n");
   printf("Lunar coordinates (J2000.0 mean ecliptic and equinox):\n");
   printf("Parameters fitted to JPL ephemeris DE405/DE406.\n");
   printf("-------------------------------------------------------\n");
   printf("JD: %f\n", JD);
   printf("X, Y, Z (km): %f   %f   %f\n\n", X,Y,Z);

   return 0;
}