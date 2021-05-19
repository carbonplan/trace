import base64
import itertools
import json
import netrc
import os
import ssl
import sys
from getpass import getpass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import dask
import fsspec

# from pangeo_forge.utils import chunked_iterable

# short_name = 'GLAH01'
# version = '033'
short_name = "GLAH14"
version = "034"
time_start = "2003-02-20T00:00:00Z"
time_end = "2009-10-11T23:59:59Z"
bounding_box = ""
polygon = ""
filename_filter = ""

CMR_URL = "https://cmr.earthdata.nasa.gov"
URS_URL = "https://urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = (
    "{0}/search/granules.json?provider=NSIDC_ECS"
    "&sort_key[]=start_date&sort_key[]=producer_granule_id"
    "&scroll=true&page_size={1}".format(CMR_URL, CMR_PAGE_SIZE)
)


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def get_username():
    username = ""

    # For Python 2/3 compatibility:
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input

    while not username:
        try:
            username = do_input("Earthdata username: ")
        except KeyboardInterrupt:
            quit()
    return username


def get_password():
    password = ""
    while not password:
        try:
            password = getpass("password: ")
        except KeyboardInterrupt:
            quit()
    return password


def get_credentials(url):
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    errprefix = ""
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators(urlparse(URS_URL).hostname)
        errprefix = "netrc error: "
    except Exception as e:
        if not ("No such file" in str(e)):
            print("netrc error: {0}".format(str(e)))
        username = None
        password = None

    while not credentials:
        if not username:
            username = get_username()
            password = get_password()
        credentials = "{0}:{1}".format(username, password)
        credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")

        if url:
            try:
                req = Request(url)
                req.add_header("Authorization", "Basic {0}".format(credentials))
                opener = build_opener(HTTPCookieProcessor())
                opener.open(req)
            except HTTPError:
                print(errprefix + "Incorrect username or password")
                errprefix = ""
                credentials = None
                username = None
                password = None

    return credentials


def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))  # Strip off any leading zeros
    query_params = ""

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += "&version={0}".format(padded_version)
        desired_pad_length -= 1
    return query_params


def build_cmr_query_url(
    short_name,
    version,
    time_start,
    time_end,
    bounding_box=None,
    polygon=None,
    filename_filter=None,
):
    params = "&short_name={0}".format(short_name)
    params += build_version_query_params(version)
    params += "&temporal[]={0},{1}".format(time_start, time_end)
    if polygon:
        params += "&polygon={0}".format(polygon)
    elif bounding_box:
        params += "&bounding_box={0}".format(bounding_box)
    if filename_filter:
        option = "&options[producer_granule_id][pattern]=true"
        params += "&producer_granule_id[]={0}{1}".format(filename_filter, option)
    return CMR_FILE_URL + params


def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if "feed" not in search_results or "entry" not in search_results["feed"]:
        return []

    entries = [e["links"] for e in search_results["feed"]["entry"] if "links" in e]
    # Flatten "entries" to a simple list of links
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if "href" not in link:
            # Exclude links with nothing to download
            continue
        if "inherited" in link and link["inherited"] is True:
            # Why are we excluding these links?
            continue
        if "rel" in link and "data#" not in link["rel"]:
            # Exclude links which are not classified by CMR as "data" or "metadata"
            continue

        if "title" in link and "opendap" in link["title"].lower():
            # Exclude OPeNDAP links--they are responsible for many duplicates
            # This is a hack; when the metadata is updated to properly identify
            # non-datapool links, we should be able to do this in a non-hack way
            continue

        filename = link["href"].split("/")[-1]
        if filename in unique_filenames:
            # Exclude links with duplicate filenames (they would overwrite)
            continue
        unique_filenames.add(filename)

        urls.append(link["href"])

    return urls


def cmr_search(
    short_name,
    version,
    time_start,
    time_end,
    bounding_box="",
    polygon="",
    filename_filter="",
):
    """Perform a scrolling CMR query for files matching input criteria."""
    cmr_query_url = build_cmr_query_url(
        short_name=short_name,
        version=version,
        time_start=time_start,
        time_end=time_end,
        bounding_box=bounding_box,
        polygon=polygon,
        filename_filter=filename_filter,
    )
    print("Querying for data:\n\t{0}\n".format(cmr_query_url))

    cmr_scroll_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        urls = []
        while True:
            req = Request(cmr_query_url)
            if cmr_scroll_id:
                req.add_header("cmr-scroll-id", cmr_scroll_id)
            response = urlopen(req, context=ctx)
            if not cmr_scroll_id:
                # Python 2 and 3 have different case for the http headers
                headers = {k.lower(): v for k, v in dict(response.info()).items()}
                cmr_scroll_id = headers["cmr-scroll-id"]
                hits = int(headers["cmr-hits"])
                if hits > 0:
                    print("Found {0} matches.".format(hits))
                else:
                    print("Found no matches.")
            search_page = response.read()
            search_page = json.loads(search_page.decode("utf-8"))
            url_scroll_results = cmr_filter_urls(search_page)
            if not url_scroll_results:
                break
            if hits > CMR_PAGE_SIZE:
                print(".", end="")
                sys.stdout.flush()
            urls += url_scroll_results

        if hits > CMR_PAGE_SIZE:
            print()
        return urls
    except KeyboardInterrupt:
        quit()


@dask.delayed
def cmr_download(urls, cache_location, credentials=None):
    """Download files from list of urls."""
    out = []
    if not urls:
        return out

    url_count = len(urls)
    print("Downloading {0} files...".format(url_count))

    for index, url in enumerate(urls, start=1):
        if not credentials and urlparse(url).scheme == "https":
            credentials = get_credentials(url)

        filename = url.split("/")[-1]
        target_url = os.path.join(cache_location, filename)
        try:
            fsspec.open(target_url).open()
            out.append(target_url)
            continue
        except FileNotFoundError:
            pass

        try:
            # In Python 3 we could eliminate the opener and just do 2 lines:
            # resp = requests.get(url, auth=(username, password))
            # open(filename, 'wb').write(resp.content)
            req = Request(url)
            if credentials:
                req.add_header("Authorization", "Basic {0}".format(credentials))
            opener = build_opener(HTTPCookieProcessor())

            with fsspec.open(target_url, mode="wb") as target:
                target.write(opener.open(req).read())

            out.append(target_url)

        except HTTPError as e:
            print("HTTPError {0}, {1}".format(e.code, e.reason), filename)
        except URLError as e:
            print("URLError: {0}".format(e.reason), filename)
        except IOError as e:
            print("IOError: {0}".format(e.reason), filename)
        except KeyboardInterrupt:
            quit()
        except:
            print("unknown error", filename)
    return out


def main():

    url_list = cmr_search(
        short_name,
        version,
        time_start,
        time_end,
        bounding_box=bounding_box,
        polygon=polygon,
        filename_filter=filename_filter,
    )

    credentials = get_credentials(url_list[0])

    tasks = []
    for url_group in chunked_iterable(url_list, 100):
        tasks.append(
            cmr_download(
                url_group,
                #                 "gs://carbonplan-scratch/glas-cache/",
                "gs://carbonplan-climatetrace/inputs/glas-raw/",
                credentials=credentials,
            )
        )

    dask.compute(tasks, retries=3, scheduler="single-threaded")


if __name__ == '__main__':
    main()
