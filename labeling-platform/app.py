import os, logging, uuid
from pathlib import Path

import streamlit as st
from streamlit.web.server.websocket_headers import _get_websocket_headers
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker, joinedload

from annotations import db
from annotations.object_detection.rect import Rects, rectchange

import annotation_utils as utils
from annotation_states import ObjectDetectionState
import config as conf

# shared queue to work with different threads
from queue import Queue
import threading

logging.basicConfig(level=logging.INFO)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Made with streamlit'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def get_db_session():
    Session = sessionmaker(bind=conf.db_engine)
    return Session()

def load_project(project_name:str):
    annotator_email = os.getenv("ANNOTATOR_EMAIL")
    with get_db_session() as db_session:
        available_projects = utils.get_projects(
            name=project_name,
            annotator_email=annotator_email,
            eager_load=True, 
            session=db_session)
    if len(available_projects) == 0:
        raise ValueError(f"No projects with name {project_name} found for annotator {annotator_email}.")
        
    return available_projects[0]

def refresh_state():
    logging.warning("Refreshing app state...")
    annotator_email = os.getenv("ANNOTATOR_EMAIL")
    with get_db_session() as db_session:
        annotator = db_session.query(db.Annotator).filter(db.Annotator.email == annotator_email).first()
        if annotator is None:
            raise ValueError(f"Annotator with email {annotator_email} not found.")
        available_projects = db_session.query(db.Project.name)\
            .join(db.ProjectAnnotator).join(db.Annotator)\
            .filter(db.Annotator.email == annotator_email)\
            .all()
        if len(available_projects) == 0:
            st.error(f"No projects found for annotator {annotator_email}.")
            return
        available_projects = [p[0] for p in available_projects]
        available_tags = [t[0] for t in db_session.query(func.distinct(db.ArtifactTag.tag)).all()]
        available_labels = [
            l[0] for l in db_session.query(func.distinct(db.AnnotationProperty.text_value))
            .where(db.AnnotationProperty.name == "label").all()
        ]
    st.session_state['session_id'] = str(uuid.uuid4())
    logging.warning(f"Session ID: {st.session_state['session_id']}")
    st.session_state['available_projects'] = available_projects
    st.session_state['available_tags'] = available_tags
    st.session_state['available_labels'] = available_labels
    st.session_state["annotator"] = annotator
    st.session_state["project"] = load_project(available_projects[0])
    st.session_state["project_artifacts"] = sorted(st.session_state["project"].artifacts, key=lambda a: a.name)
    st.session_state["not_annotated_inds"] = [
        i for i,art in enumerate(st.session_state["project_artifacts"]) if len(art.annotations) == 0]
    st.session_state["artifact_index"] = 0
    st.session_state["automatic_annotations"] = {}
    st.session_state['previous_rects'] = None
    st.session_state['files'] = [a.uri for a in st.session_state["project_artifacts"]]

def save_session_state_to_db():
    logging.warning("Saving app session state to db...")
    with get_db_session() as db_session:
        annotation_state_model = ObjectDetectionState(
            **st.session_state,
            artifact=st.session_state["project_artifacts"][st.session_state["artifact_index"]],
        )
        annotation_state_json = annotation_state_model.dict()
        logging.info(f"Saving annotation state:\n{annotation_state_json}")
        anno_sess_state = db.SessionState(
            session_id=st.session_state['session_id'],
            annotator_id=st.session_state['annotator'].id,
            project_id=st.session_state['project'].id,
            session_state=annotation_state_json
        )
        db_session.add(anno_sess_state)
        db_session.commit()

if "project" not in st.session_state.keys():
    refresh_state()

project = st.session_state["project"]
available_projects = st.session_state["available_projects"]

available_tags = st.session_state["available_tags"]
file_paths = st.session_state["files"]
n_files = len(st.session_state["project_artifacts"])
project_artifacts = st.session_state["project_artifacts"]
annotated = [a for a in project_artifacts if len(a.annotations) > 0]

# sidebar
with st.sidebar:
    st.write("Files in project:", n_files)
    st.write("Filtered files:", len(st.session_state["files"]))
    st.write("Total annotated files:", len(annotated))
    # st.write("Remaining files:", n_files - len(annotated))

    # select annotation project
    def change_project():
        if proj_name != project.name:
            st.write("Selected labeling project:", proj_name)
            # change project based on selected annotation task
            project = load_project(proj_name)
            st.session_state["project"] = project
            st.session_state["project_artifacts"] = sorted(project.artifacts, key=lambda a: a.name)
    
    proj_name = st.selectbox(
        "Annotation Project",
        options=available_projects,
        index=0,
        key="project_name",
        on_change=change_project
    )
    
    # select image to annotate
    logging.info(f'Image index: { st.session_state["artifact_index"]}')
    
    def go_to_image():
        file_index = [i for i,f in enumerate(st.session_state['files']) if f == st.session_state["file"]]
        if len(file_index) > 0:
            file_index = file_index[0]
        else:
            file_index = 0
        st.session_state["artifact_index"] = file_index

    selected_img = st.selectbox(
        "Files",
        options=file_paths,
        index=st.session_state["artifact_index"],
        on_change=go_to_image,
        key="file",
        format_func=lambda f: Path(f).stem,
    )
    logging.info(f'Selected image: { file_paths[st.session_state["artifact_index"]]}')
    st.session_state["currect_artifact"] = project_artifacts[st.session_state["artifact_index"]]

    # select model for auto-annotation
    anno_labels = conf.ANNOTATION_TASKS[project.name]
    available_annotation_models = [a.name for a in project.annotators if a.automatic]
    if available_annotation_models:
        model_name = st.selectbox(
            "ML Model for Auto-Annotation",
            options=[m for m in available_annotation_models],
            index=0,
        )
        st.write("Selected model:", model_name)
        st.session_state["selected_annotation_model"] = model_name
        st.session_state["selected_model_info"] = conf.MODEL_INFO[model_name]

    # filter by artifact labels 
    if len(available_tags) > 0:
        def filter_by_tags():
            logging.info(f'selected tags { st.session_state["tags"]}')
            # label box filter with all labels available by default
            st.session_state['files'] = [
                a.uri for a in st.session_state["project_artifacts"]
                if any([l.tag in st.session_state['tags'] for l in a.tags]) or 
                (len(a.tags) == 0 and st.session_state['show_untagged'])
            ]
            st.session_state['file'] = st.session_state['files'][0]
            go_to_image()
        logging.info(f"Available tags: {available_tags}")
        st.multiselect(
            "Tags",
            options=available_tags,
            default=available_tags,
            key="tags",
            on_change=filter_by_tags,
        )
        # slider wether to also show artifacts without tags
        st.checkbox("Show artifacts without tags", value=True, key="show_untagged", on_change=filter_by_tags)

    # uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    # Sidebar buttons to navigate images
    col1, col2 = st.columns(2)
    with col1:
        def previous_image():
            image_index = st.session_state["artifact_index"]
            if image_index <= 0:
                st.toast("This is the first image.", icon="⚠️")
                return
            st.session_state["artifact_index"] -= 1
            st.session_state["previous_rects"] = None

        st.button(label="Previous image", on_click=previous_image)
    with col2:
        def next_image():
            image_index = st.session_state["artifact_index"]
            if image_index >= len(file_paths) - 1:
                st.toast("This is the last image.", icon="⚠️")
            st.session_state["artifact_index"] += 1
            st.session_state["previous_rects"] = None

        st.button(label="Next image", on_click=next_image)

    def next_annotate_file():
        # TODO: get this from active learning model
        image_index = st.session_state["artifact_index"]
        if len(st.session_state["not_annotated_inds"]) == 0:
            st.warning("All images are annotated.")
            next_image()
        # get closest not annotated image
        next_image_index = st.session_state["not_annotated_inds"][0]
        logging.info(f'Next image index: { next_image_index}')
        for i in range(image_index+1, len(file_paths)):
            if i in st.session_state["not_annotated_inds"]:
                next_image_index = i
                break
        st.session_state["artifact_index"] = next_image_index

    st.button(label="Next need annotate", on_click=next_annotate_file)

    # button to refresh session
    st.button(label="Refresh", on_click=refresh_state)

    st.write(f"Session id: `{st.session_state['session_id']}`")

# Main content: annotate images:

# prepare image and annotation rects
selected_artifact = next(filter(lambda a: a.uri == selected_img, project_artifacts), None)
if selected_artifact is None:
    st.error("No image selected.")
    st.stop()

logging.info(f"Selected artifact: {selected_artifact}")
logging.info(f"Artifact has {len(selected_artifact.annotations)} annotations:")
img_path = os.path.join(conf.WORKING_DIR, selected_artifact.uri)
im = ImageManager(img_path, annotations_dir=conf.ANNOTATIONS_DIR)
db_rects = utils.parse_rects_from_annotations(selected_artifact.annotations)
auto_annotated_rects = st.session_state["automatic_annotations"].get(selected_artifact.name, [])
artifact_rects = db_rects + auto_annotated_rects
rects = [r.todict() for r in artifact_rects]
# if os.environ.get("RESIZED_IMG",'false').lower().startswith(('t','1')):
#     # resize image and rects if necessary
#     img =  im.resizing_img(1224, 1632) 
#     rects_ = [im._resize_rect(r) for r in rects]
# else:
# img = im._img
# rects_ = rects

img =  im.resizing_img(700, 700)
rects_ = [im._resize_rect(r) for r in rects]

logging.warning(f"Image size: {img.size}")

# show image and annotations
# logging.info(f"rects {rects}")
st_rects = st_img_label(img, box_color="red", rects=rects_, stroke_width=50)
# logging.info(f"st_rects {st_rects}")
new_rects = [im._unresize_rect(r) for r in st_rects]

# check if rects have 
curr_rects = Rects(new_rects, image_index=st.session_state["artifact_index"])
st.session_state["current_rects"] = curr_rects
if st.session_state["previous_rects"] is None:
    st.session_state["previous_rects"] = db_rects
    # save_session_state_to_db()
prev_rects = st.session_state["previous_rects"]
change_in_rects = rectchange(prev_rects, curr_rects)
if change_in_rects.has_changes:
    # save_session_state_to_db()
    pass
else:
    logging.warning("No changes in rects.")
        
# Auto-annotate button and button to save annotations
col1, col2 = st.columns(2, gap="small")
with col1:
    # function to get annotations from model
    def auto_annotate():
        if (model:=st.session_state["selected_model_info"]):
            try:
                auto_rects = utils.get_rects_from_model(
                    model.get("uri"), im.get_img(), 
                    label=st.session_state['available_labels'][0].title(),
                    annotator_name=model.get("name"),
                    annotator_automatic=True)
            except Exception as e:
                st.error(f"Error while getting annotations from model: {e}")
                logging.error(e)
                st.stop()
            logging.info(f"{len(auto_rects)} object detected.")
            st.session_state["automatic_annotations"][selected_artifact.name] = auto_rects
            if len(auto_rects)>0:
                st.toast(f"{len(auto_rects)} objects were automatically annoted.", icon="✅")
            else:
                st.toast("No objects were detected in this image.", icon="⚠️")
        else:
            st.toast("No model is selected.")

    if available_annotation_models:
        st.button(label="Auto-annotate  🪄", on_click=auto_annotate)
with col2:
    # function to save annotations
    def save_annotations():
        """ 
        Save new annotations to the database if there are any changes.
        """
        prev_rects = st.session_state["previous_rects"]
        curr_rects = st.session_state["current_rects"]
        diff = rectchange(prev_rects, curr_rects)
        if diff.has_changes:
            logging.info(f"Saving rects with {len(diff.changes)} changes.")
            save_session_state_to_db()
            st.toast("Annotations saved.", icon="✅")
        else:
            logging.info("No changes in rects.")
            st.toast("No changes in annotations to save.", icon="⚠️")
    st.button(label="Save Annotations", on_click=save_annotations)

# show image tags
if selected_artifact.tags:
    tags_str = ', '.join([t.tag for t in selected_artifact.tags])
else:
    tags_str = "[No tags]"
with (expander:=st.expander(f"Image tags: :red[{tags_str}]")):
    # function to add tags to the image in the database
    def add_tags_to_artifact(artifact:db.Artifact, tags:list):
        current_tags = [t.tag for t in selected_artifact.tags]
        logging.warning(f"Adding tags {tags} to artifact. Current tags: {current_tags}")
        with get_db_session() as db_session:
            for tag in tags:
                if tag in current_tags:
                    st.toast(f"Tag '{tag}' already exists.", icon="⚠️")
                    continue
                db_session.add(db.ArtifactTag(tag=tag, artifact_id=artifact.id))
            db_session.commit()
            st.session_state["project"] = load_project(st.session_state["project"].name)
    
    if new_tags:=st.text_input("Add tags (separated by comma):"):
        add_tags_to_artifact(selected_artifact, tags=new_tags.split(","))

# show image annotations along with their properties
anno_labels = st.session_state['available_labels']
with st.expander("Image annotations", expanded=True):
    preview_imgs = im.init_annotation(st_rects)
    logging.info(f"There are {len(im._current_rects)} annotations in the image.")
    col1, col2 = st.columns(2)
    col1.write("Preview")
    col2.write("Properties")
    for i, prev_img in enumerate(preview_imgs):
        col1, col2 = st.columns(2)
        prev_img[0].thumbnail((200, 200))
        with col1:
            col1.image(prev_img[0])
        with col2:
            default_index = 0
            if prev_img[1]:
                try:
                    default_index = anno_labels.index(prev_img[1])
                except ValueError:
                    logging.info(f"'{prev_img[1]}' not found in the list of labels ({anno_labels}).")
            select_label = col2.selectbox("Label", anno_labels, key=f"label_{i}", index=default_index)
            im.set_annotation(i, select_label)
        # separator
        
