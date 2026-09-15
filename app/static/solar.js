/* Sun-position subset adapted from SunCalc 1.9.0 (BSD-2-Clause).
 * Copyright (c) 2014, Vladimir Agafonkin. See vendor/SunCalc-LICENSE.txt.
 * https://github.com/mourner/suncalc/tree/v1.9.0
 */
window.PrismSolar = (() => {
  const RAD = Math.PI / 180;
  const DAY_MS = 86400000;

  // UTC epoch arithmetic avoids local-calendar, date-line and DST ambiguities.
  function isDaylight(date, latitude, longitude) {
    const days = date.valueOf() / DAY_MS - 0.5 + 2440588 - 2451545;
    const anomaly = RAD * (357.5291 + 0.98560028 * days);
    const center = RAD * (1.9148 * Math.sin(anomaly)
      + 0.02 * Math.sin(2 * anomaly) + 0.0003 * Math.sin(3 * anomaly));
    const ecliptic = anomaly + center + RAD * 102.9372 + Math.PI;
    const obliquity = RAD * 23.4397;
    const declination = Math.asin(Math.sin(ecliptic) * Math.sin(obliquity));
    const rightAscension = Math.atan2(Math.sin(ecliptic) * Math.cos(obliquity), Math.cos(ecliptic));
    const hourAngle = RAD * (280.16 + 360.9856235 * days + longitude) - rightAscension;
    const phi = RAD * latitude;
    const altitude = Math.asin(Math.sin(phi) * Math.sin(declination)
      + Math.cos(phi) * Math.cos(declination) * Math.cos(hourAngle));
    // Conventional sunrise/sunset: upper solar limb, with atmospheric refraction.
    // Evaluating altitude also handles polar day/night without invalid rise times.
    return altitude >= -0.833 * RAD;
  }

  return { isDaylight };
})();
