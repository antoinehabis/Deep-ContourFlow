from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
import logging

import sys
from argparse import ArgumentParser
from shapely import wkt
from shapely.affinity import affine_transform
import pandas as pd
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, ImageInstanceCollection
from cytomine.models import TermCollection



def get_by_id(haystack, needle):
    return next((item for item in haystack if item.id == needle), None)


pb_key = os.getenv('CYTOMINE_PUBLIC')
pv_key = os.getenv('CYTOMINE_PRIVATE')
host = os.getenv('CYTOMINE_HOST')

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)

project_id = os.getenv('PROJECT_ID')

with Cytomine(host=host, public_key=pb_key, private_key=pv_key) as cytomine:
    terms = TermCollection().fetch_with_filter("project", project_id)
    terms_dict = {t.id: t.name for t in terms}
    image_instances = ImageInstanceCollection().fetch_with_filter("project", project_id)
    dic_id_img = {}
    for image in image_instances:
        dic_id_img[image.id] = image.filename.split("/")[-1]
    # We want all annotations in a given project.
    annotations = AnnotationCollection()
    annotations.project = project_id  # Add a filter: only annotations from this project
    # You could add other filters:
    # annotations.image = id_image => Add a filter: only annotations from this image
    # annotations.images = list(dic_id_img.keys())
    # annotations.user = id_user => Add a filter: only annotations from this user
    # ...
    annotations.showWKT = True  # Ask to return WKT location (geometry) in the response
    annotations.showMeta = (
        True  # Ask to return meta information (id, ...) in the response
    )
    annotations.showGIS = (
        True  # Ask to return GIS information (perimeter, area, ...) in the response
    )
    annotations.showTerm = True
    # ...
    # => Fetch annotations from the server with the given filters.
    annotations.fetch()

df = pd.DataFrame(
    columns=["id", "slide", "project", "term", "area", "perimeter", "location"]
)

for annotation in annotations:
    if len(annotation.term) > 0:
        df = df.append(
            {
                "id": annotation.id,
                "slide": dic_id_img[annotation.image],
                "project": annotation.project,
                "term": terms_dict[annotation.term[0]],
                "area": annotation.area,
                "perimeter": annotation.perimeter,
                "location": annotation.location,
            },
            ignore_index=True,
        )
df.to_csv(os.path.join(path_annotations, "annotations.csv"))
