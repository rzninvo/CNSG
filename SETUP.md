# CNSG Installation and Setup Guide

## Clone the Repository

```bash
git clone https://github.com/rzninvo/CNSG.git
```


## 1. Preparing Conda Environment

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda environment:

```bash
# We require python>=3.9 and cmake>=3.10
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```

---

## 2. Install habitat-sim

- **[Laptop]** To install habitat-sim on a system with display:
  ```bash
  conda install habitat-sim withbullet -c conda-forge -c aihabitat
  ```

- **[Cluster]** To install habitat-sim on a headless system:
  ```bash
  conda install habitat-sim withbullet headless -c conda-force -c aihabitat
  ```

> The `withbullet` parameter is optional (but recommended).

---

## Testing

### 1. Download 3D Assets

Let's download some 3D assets using the python data download utility:

```bash
cd habitat-sim
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ./data
```

---

### 2. Interactive Testing (e.g., laptop)

Use the interactive viewer included with Habitat-Sim in either C++ or Python:

```bash
# NOTE: depending on your choice of installation, you may need to add '/path/to/habitat-sim' to your PYTHONPATH.
# e.g. from 'habitat-sim/' directory run 'export PYTHONPATH=$(pwd)'
# inside 'habitat-sim/' folder
export PYTHONPATH=$(pwd)
python examples/viewer.py --scene ./data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```

You should be able to control an agent in this test scene.  
Use **W/A/S/D** keys to move forward/left/backward/right and arrow keys or mouse (LEFT click) to control gaze direction (look up/down/left/right).

---

### 3. Non-interactive Testing (headless systems)

**[IGNORE if on laptop]** Run the example script:

```bash
python /path/to/habitat-sim/examples/example.py --scene /path/to/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```

The agent will traverse a particular path and you should see performance stats at the end, e.g.:  
`640 x 480, total time: 3.208 sec. FPS: 311.7`

To reproduce the benchmark table from [Habitat ICCV'19](https://arxiv.org/abs/1904.01201), run:

```bash
examples/benchmark.py --scene /path/to/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb
```

Additional arguments to `example.py` allow you to change the sensor configuration, print statistics of semantic annotations, compute shortest-path trajectories, and more.  
Refer to `example.py` and `demo_runner.py` source files for an overview.

To load a specific MP3D or Gibson house:
```bash
examples/example.py --scene path/to/mp3d/house_id.glb
```

We have also provided an [example demo](https://aihabitat.org/docs/habitat-lab/habitat-lab-demo.html) for reference.

To run a physics example in Python (after building with "Physics simulation via Bullet"):

```bash
python examples/example.py --scene /path/to/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb --enable_physics
```

In this mode, the agent will be frozen and oriented toward the spawned physical objects.  
Additionally, `--save_png` can be used to output visual observation frames to the current directory.

---

> These instructions should be sufficient. In case of errors, more detailed instructions are available in `/habitat-sim/README.md`.

---

## Download Habitat-Matterport 3D Research Dataset (HM3D)

### Downloading HM3D with the Download Utility

First, you need to generate a Matterport API Token:

1. Navigate to [https://my.matterport.com/settings/account/devtools](https://my.matterport.com/settings/account/devtools)
2. Generate an API token
3. Your **API token ID** functions as your username (`--username`), and your **API token secret** functions as your password (`--password`).  
   > Make sure to write your API token secret down — it cannot be revealed again!

Now, you are ready to download. To download the **minival split**:

```bash
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```

> These instructions should be sufficient. In case of errors, more detailed instructions are available in `/habitat-sim/DATASETS.md`.

---

## Run a Scene from HM3D

In the Habitat conda environment (`conda activate habitat`, if not active):

```bash
cd habitat-sim
python examples/mr_viewer.py   --dataset "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"   --scene "data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
```