from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import sys
from argparse import ArgumentParser

from shapely.geometry import Point, box

from cytomine import Cytomine
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection
import os

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)


def delete_annotations(id_image, id_project):
    pb_key = os.getenv('CYTOMINE_PUBLIC')
    pv_key = os.getenv('CYTOMINE_PRIVATE')
    host = os.getenv('CYTOMINE_HOST')

    with Cytomine(host=host, public_key=pb_key, private_key=pv_key) as cytomine:
        # Get the list of annotations
        annotations = AnnotationCollection()
        annotations.image = id_image
        annotations.project = id_project
        annotations.fetch()
        for annotation in annotations:
            annotation.delete()
    return "You deleted all the annnotations"
