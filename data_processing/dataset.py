from nova_utils.data.provider.data_manager import DatasetManager
from dotenv import load_dotenv
import os
import numpy as np
import h5py
from pathlib import Path

def load_environment_variables():
    load_dotenv(".env")
    return {
        "ip": os.getenv("NOVA_IP", ""),
        "port": int(os.getenv("NOVA_PORT", 0)),
        "user": os.getenv("NOVA_USER", ""),
        "password": os.getenv("NOVA_PASSWORD", ""),
        "data_dir": os.getenv("NOVA_DATA_DIR", None),
    }

def build_data_description(roles):
    data_description = [{
        "id": "session_w2v_bert",
        "src": "db:stream:SSIStream",
        "type": "input",
        "role": "session",
        "name": "w2v_bert_2_embeddings",
    }]

    for role in roles:
        data_description.extend([
            {
                "id": f"{role}_icep",
                "src": "db:annotation:Discrete",
                "type": "input",
                "role": role,
                "scheme": f"{role.title()}Engagement",
                "annotator": "gold",
            },
            {
                "id": f"{role}_dino",
                "src": "db:stream:SSIStream",
                "type": "input",
                "role": role,
                "name": "dino_v2_embeddings",
            }
        ])
    return data_description

def sample_annotations(annotations, frame_rate):
    frame_duration = 1000 // frame_rate
    new_annotations = []
    dtype = annotations.dtype.descr
    last_end_time = 0

    for annotation in annotations:
        start_time, end_time, annotation_type, confidence = annotation

        while last_end_time < start_time:
            gap_end_time = min(start_time, last_end_time + frame_duration)
            new_annotations.append((last_end_time, gap_end_time, -1, 1.0))
            last_end_time = gap_end_time

        while start_time < end_time:
            next_time = min(end_time, start_time + frame_duration)
            new_annotations.append((start_time, next_time, annotation_type, confidence))
            start_time = next_time

        last_end_time = end_time

    return np.array(new_annotations, dtype=dtype)

def sample_anno_to_labels(annotations, frame_rate, anno_classes):
    frame_duration = 1000 // frame_rate
    labels = []

    last_end_time = 0

    for annotation in annotations:
        start_time, end_time, annotation_type, _ = annotation

        while last_end_time < start_time:
            gap_end_time = min(start_time, last_end_time + frame_duration)
            labels.append("NoAnno")
            last_end_time = gap_end_time

        while start_time < end_time:
            next_time = min(end_time, start_time + frame_duration)
            label = anno_classes.get(annotation_type, {'name': 'Garbage'})['name']
            labels.append(label)
            start_time = next_time

        last_end_time = end_time

    return np.array(labels)
# def save_session_data_hdf5(session_w2v_bert, infant_dino, caretaker_dino, labels_infant, labels_caretaker, session_file_path):
#     session_dir = Path(session_file_path).parent
#     session_dir.mkdir(parents=True, exist_ok=True)
#     hdf5_file_path = session_dir / f"w2v_dino_labels_session_data.hdf5"
#
#     with h5py.File(hdf5_file_path, "w") as f:
#         f.create_dataset("w2v_features", data=session_w2v_bert)
#         f.create_dataset("infant_dino_features", data=infant_dino)
#         f.create_dataset("caretaker_dino_features", data=caretaker_dino)
#         f.create_dataset("infant_labels", data=np.array(labels_infant, dtype='S'))
#         f.create_dataset("caretaker_labels", data=np.array(labels_caretaker, dtype='S'))
#     print(f"Saved session data to {hdf5_file_path}")
def save_session_data_hdf5(session_w2v_bert, infant_dino, caretaker_dino,
                           labels_infant, labels_caretaker,
                           session_name, session_file_path):
    session_dir = Path(session_file_path).parent
    session_dir.mkdir(parents=True, exist_ok=True)
    previous_path = session_dir / f"w2v_dino_labels_session_data.hdf5"
    if previous_path.exists():
        previous_path.unlink()
    hdf5_file_path = session_dir / f"{session_name}_w2v_dino_labels_data.hdf5"

    # all arrays are not the same length:
    # determine the number of frames based on the smallest length among features and labels
    num_frames = min(len(session_w2v_bert), len(infant_dino), len(caretaker_dino),
                     len(labels_infant), len(labels_caretaker))

    session_names = np.array([session_name.encode('utf-8')] * num_frames)
    frame_numbers = np.arange(num_frames)

    with h5py.File(hdf5_file_path, "w") as f:
        f.create_dataset("frame_numbers", data=frame_numbers)
        f.create_dataset("session_names", data=session_names)
        f.create_dataset("w2v_features", data=session_w2v_bert[:num_frames])
        f.create_dataset("infant_dino_features", data=infant_dino[:num_frames])
        f.create_dataset("caretaker_dino_features", data=caretaker_dino[:num_frames])
        f.create_dataset("infant_labels", data=np.array(labels_infant[:num_frames], dtype='S'))
        f.create_dataset("caretaker_labels", data=np.array(labels_caretaker[:num_frames], dtype='S'))


    print(f"Saved session data to {hdf5_file_path}")


def collect_session_data(dataset_manager):
    w2v_features = []
    dino_features_infant = []
    dino_features_caretaker = []
    labels_infant = []
    labels_caretaker = []
    for session_name, session_dict in dataset_manager.sessions.items():
        session_manager = session_dict['manager']
        session_manager.load()
        data = session_manager.input_data

        session_file_path = session_manager.input_data['session_w2v_bert'].meta_data.file_path

        # Extract features and labels
        session_w2v_bert = data['session_w2v_bert'].data
        infant_dino = data['infant_dino'].data
        caretaker_dino = data['caretaker_dino'].data
        infant_icep = data['infant_icep']
        caretaker_icep = data['caretaker_icep']

        #sampled_infant_icep = sample_annotations(infant_icep.data, data['session_w2v_bert'].meta_data.sample_rate)
        #sampled_caretaker_icep = sample_annotations(caretaker_icep.data, data['session_w2v_bert'].meta_data.sample_rate)

        infant_anno_classes = infant_icep.annotation_scheme.classes
        caretaker_anno_classes = caretaker_icep.annotation_scheme.classes

        labels_infant = sample_anno_to_labels(infant_icep.data, data['session_w2v_bert'].meta_data.sample_rate, infant_anno_classes)
        labels_caretaker = sample_anno_to_labels(caretaker_icep.data, data['session_w2v_bert'].meta_data.sample_rate, caretaker_anno_classes)



        w2v_features.append(session_w2v_bert)
        dino_features_infant.append(infant_dino)
        dino_features_caretaker.append(caretaker_dino)

        save_session_data_hdf5(session_w2v_bert, infant_dino, caretaker_dino,
                              labels_infant, labels_caretaker,session_name,
                              session_file_path)


def main():
    config = load_environment_variables()

    db_context = {
        "db": {
            "db_host": config["ip"],
            "db_port": config["port"],
            "db_user": config["user"],
            "db_password": config["password"],
            "data_dir": config["data_dir"],
        },
    }

    dataset_name = 'DFG_A1_A2b'
    roles = ['infant', 'caretaker']
    data_description = build_data_description(roles)

    dataset_manager = DatasetManager(dataset=dataset_name, data_description=data_description, source_context=db_context)

    collect_session_data(dataset_manager)

if __name__ == "__main__":
    main()
