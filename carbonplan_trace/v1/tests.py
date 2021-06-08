import numpy as np
import utm

# create utm band letter / latitude dictionary
# latitude represents southern edge of letter band
band_numbers = list(np.arange(-80,80,8))
band_numbers.append(84)

def spans_utm_border(lats):
    '''
    find if a latitude range of a scene spans more than 1 band
    '''
    min_lat = np.min(np.array(lats))
    max_lat = np.max(np.array(lats))
    
    if ((band_numbers < max_lat).astype(int) + \
        (band_numbers > min_lat).astype(int) != 1).sum():
        # this logic only evaluates if the lats span more than
        # one interval in the lat bands
        return True
    else:
        return False


def test_proj(coords, projected, zone_number):
    '''
    Use UTM to project a provided lat/lon coordinate into
    x/y space and see if they match.
    If they do, grab a letter. If not, grab the other letter (and
    confirm that it also works?)
    '''
    # test out for a given coordinate 
    (test_x, test_y, calculated_zone_number, calculated_zone_letter) = utm.from_latlon(coords[1], coords[0], force_zone_number=zone_number)
    tolerance = 2 # in meters - should really be within 0.5 meters
    # These will fail if the test latlon-->meters projection was off by more than 
    # 2 meters from the values provided in the metadata
    assert abs(test_x-projected[0])<tolerance
    assert abs(test_y-projected[1])<tolerance
    return calculated_zone_letter