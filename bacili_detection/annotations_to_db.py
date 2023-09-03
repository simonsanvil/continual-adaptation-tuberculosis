"""
Add annotations used by Visuña et al. 2023 to the database
with the appropriate train/validation split tags
"""

import os, re, dotenv
import glob
from pathlib import Path

from sqlalchemy.orm import joinedload
from annotations import db

dotenv.load_dotenv('.env')

db_session = db.get_session(os.environ['DATABASE_URI'])

# get all the images
train_images = glob.glob('data/visuna_2023/train/tuberculosis-*.jpg')
test_images = glob.glob('data/visuna_2023/test/tuberculosis-*.jpg')
val_images = glob.glob('data/visuna_2023/val/val-tuberculosis-*.jpg')

# how many images are there?
print("Images in directories:")
print(f"\tTrain images: {len(train_images)}")
print(f"\tTest images: {len(test_images)}")
print(f"\tVal images: {len(val_images)}")

# images in the db
artifacts = (
    db_session.query(db.Artifact)
    .join(db.Project)
    .options(joinedload(db.Artifact.tags))
    .where(db.Project.name == "Bacilli Detection")
)
# how many images have train tags?
print("In DB:")
train_artifacts = [art for art in artifacts if any("train"==t.tag for t in art.tags)]
print(f"\tTrain images: {len(train_artifacts)}")
test_artifacts = [art for art in artifacts if any("test"==t.tag for t in art.tags)]
print(f"\tTest images: {len(test_artifacts)}")
val_artifacts = [art for art in artifacts if any("val"==t.tag for t in art.tags)]
print(f"\tVal images: {len(val_artifacts)}")

# how many of them coincide?
def extract_image_id(path:str) -> int:
    s = re.search(r'.*tuberculosis-phone-(\d+)', Path(path).stem)
    if s is None:
        raise ValueError(f"Could not extract image id from {path}")
    return s.group(1)

artifact_ids_dict = {
    extract_image_id(art.uri): art
    for art in artifacts
}

train_art_img_ids = {imid for imid, art in artifact_ids_dict.items() if art in train_artifacts}
test_art_img_ids = {imid for imid, art in artifact_ids_dict.items() if art in test_artifacts}
val_art_img_ids = {imid for imid, art in artifact_ids_dict.items() if art in val_artifacts}
local_train_img_ids = {extract_image_id(path) for path in train_images}
local_test_img_ids = {extract_image_id(path) for path in test_images}
local_val_img_ids = {extract_image_id(path) for path in val_images}

print("In DB and in directories:")
print(f"\tTrain images: {len(train_art_img_ids.intersection(local_train_img_ids))}")
print(f"\tTest images: {len(test_art_img_ids.intersection(local_test_img_ids))}")
print(f"\tVal images: {len(val_art_img_ids.intersection(local_val_img_ids))}")

# add the corresponding tags to the arts that are tagged in the directories but not in the db
def tag_image_ids(img_ids, tag):
    new_tags = []
    for img_id in img_ids:
        if img_id not in artifact_ids_dict:
            print(f"Image with id {img_id} (tuberculosis-phone-{img_id}) not in database")
            continue
        art = artifact_ids_dict[img_id]
        art_tag = db.ArtifactTag(tag=tag, artifact_id=art.id)
        new_tags.append(art_tag)
    db_session.add_all(new_tags)
    db_session.commit()
    print(f"Succesfully tagged {len(new_tags)} artifacts in the db with tag \"{tag}\"")

tag_image_ids(local_train_img_ids.difference(train_art_img_ids), "train")
tag_image_ids(local_test_img_ids.difference(test_art_img_ids), "test")
tag_image_ids(local_val_img_ids.difference(val_art_img_ids), "val")