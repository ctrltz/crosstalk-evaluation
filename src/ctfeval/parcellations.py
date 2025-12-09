import mne
import numpy as np

from dataclasses import dataclass, field


@dataclass
class Parcellation:
    fs_name: str
    code: str
    name: str
    remove_unknown: callable
    labels: list = field(default_factory=list)
    label_names: list = field(default_factory=list)
    n_labels: int = 0
    loaded: bool = False

    def __getitem__(self, name):
        return self.labels[self.index(name)]

    def index(self, name):
        return self.label_names.index(name)

    def areas(self, *args, **kwargs):
        return np.array([label.compute_area(*args, **kwargs) for label in self.labels])

    def load(self, subject, subjects_dir, remove_unknown=True):
        if self.loaded:
            return self

        self.labels = mne.read_labels_from_annot(
            subject, parc=self.fs_name, subjects_dir=subjects_dir
        )

        # Remove unmapped areas on the medial wall
        if remove_unknown:
            self.labels = self.remove_unknown(self.labels)

        self.n_labels = len(self.labels)
        self.label_names = [label.name for label in self.labels]
        self.loaded = True

        return self


_parcellations = [
    # anatomical
    Parcellation(
        fs_name="aparc",
        code="DK",
        name="Desikan-Killiany",
        remove_unknown=lambda p: p[:-1],
    ),
    Parcellation(
        fs_name="aparc.a2009s",
        code="Destrieux",
        name="Destrieux",
        remove_unknown=lambda p: p[:-2],
    ),
    # cytoarchitectonic
    Parcellation(
        fs_name="PALS_B12_Brodmann",
        code="BA",
        name="Brodmann",
        remove_unknown=lambda p: [label for label in p if "Brodmann" in label.name],
    ),
    # functional
    Parcellation(
        fs_name="Schaefer2018_400Parcels_17Networks_order",
        code="Schaefer400",
        name="Schaefer (400 ROIs)",
        remove_unknown=lambda p: p[:-2],
    ),
    # multimodal
    Parcellation(
        fs_name="HCPMMP1", code="HCP", name="HCP", remove_unknown=lambda p: p[2:]
    ),
    Parcellation(
        fs_name="HCPMMP1_combined",
        code="HCP_combined",
        name="HCP (combined)",
        remove_unknown=lambda p: p[2:],
    ),
]
PARCELLATIONS = {p.code: p for p in _parcellations}
