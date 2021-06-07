import numpy as np
import utm

# create utm band letter / latitude dictionary
# latitude represents southern edge of letter band

band_letters = ['C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
band_numbers = list(np.arange(-80,80,8))
band_numbers.append(84)
utm_band_dict = dict(zip(band_letters, band_numbers))

def calculate_zone_letter(lat):
    '''
    Given a latitude calculate what the zone letter is
    '''
    zone_letter_index_location = ((np.array(band_numbers) - lat) <= 0).sum()
    return band_letters[zone_letter_index_location]
    
def spans_utm_border(lats):
    '''
    find if a latitude range of a scene spans more than 1 band
    '''
    print(lats)
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
    # print(utm.from_latlon(coords[1], coords[0], force_zone_number=zone_number))
    (test_x, test_y, calculated_zone_number, calculated_zone_letter) = utm.from_latlon(coords[1], coords[0], force_zone_number=zone_number)
    tolerance = 2 # in meters - should really be within 0.5 meters
    # print('{} is landsat provided x and {} is test projected x'.format(projected[0], test_x))
    # print('{} is landsat provided y and {} is test projected y'.format(projected[1], test_y))
    # print('{} is landsat provided zone number and {} is test calculated zone number'.format(zone_number, calculated_zone_number))
    # print(test_x)
    
    # These will fail if the test latlon-->meters projection was off by more than 
    # 2 meters from the values provided in the metadata
    assert abs(test_x-projected[0])<tolerance
    assert abs(test_y-projected[1])<tolerance
    return calculated_zone_letter