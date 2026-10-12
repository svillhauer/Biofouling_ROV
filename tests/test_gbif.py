from biofouling.gbif import recover_url_and_id


def test_well_formed_row():
    fields = ["1234567890", "StillImage", "image/jpeg", "https://inaturalist.org/photos/1/large.jpg"]
    assert recover_url_and_id(fields, "row0") == ("https://inaturalist.org/photos/1/large.jpg", "1234567890")


def test_ragged_row_with_shifted_columns():
    # An undeclared datasetKey pushes every column right by one.
    fields = ["50c9509d-22c7-4a22-a47d-8c48425ef4a7", "987654321", "StillImage", "",
              "http://example.org/media/IMG_0042.JPEG?size=large"]
    url, gbif_id = recover_url_and_id(fields, "row7")
    assert url == "http://example.org/media/IMG_0042.JPEG"
    assert gbif_id == "987654321"


def test_missing_values_fall_back():
    url, gbif_id = recover_url_and_id(["StillImage", "no url here"], "row3")
    assert url is None
    assert gbif_id == "row3"
