# CONG dataset

This repository provides the code for creating the CONG dataset from the paper [Constrained Generative Sampling of 6-DoF Grasps](https://arxiv.org/pdf/2302.10745.pdf). 

## Usage

### Pre-requisites

Create the conda environment:
```bash
conda env create -f environment.yml
```

### CLI Options

Run the script using the command `python main.py` with the following options:

- `-mp, --mesh_path`: Path to meshes (default: `data/shapenetsem_example_meshes/`)
- `-gp, --grasp_path`: Path to acronym grasps (default: `data/acronym_example_grasps/`)
- `-sp, --folder_for_storing`: Path where to save results (default: `/tmp/constrained_grasping_dataset/`)
- `-np, --number_of_points_to_sample_on_mesh`: Number of points to sample on the mesh (default: 1024)
- `-th, --threshold`: Threshold between center of grasps and query points (default: 0.002)
- `-nq, --num_query_points`: Number of query points to sample from the mesh (default: 50)
- `-v, --visualize`: Flag to visualize the results

Example:

```bash
python main.py -v
```

## Generate the complete dataset

### Pre-requisites

1. Download the full acronym dataset: [acronym.tar.gz](https://drive.google.com/file/d/1zcPARTCQx2oeiKk7a-wdN_CN-RUVX56c/view?usp=sharing)
2. Download the ShapeNetSem meshes from [https://www.shapenet.org/](https://www.shapenet.org/)

### Usage

```bash
python main.py -mp path/to/ShapeNetSem/meshes -gp path/to/Acronym/dataset
```

## Contribution

Feel free to open issues, suggest enhancements, or make pull requests.

## Citation

If this code is useful in your research, please consider citing:

```
@article{lundell2023constrained,
  title={Constrained generative sampling of 6-dof grasps},
  author={Lundell, Jens and Verdoja, Francesco and Le, Tran Nguyen and Mousavian, Arsalan and Fox, Dieter and Kyrki, Ville},
  journal={arXiv preprint arXiv:2302.10745},
  year={2023}
}
```

## License

[MIT License](LICENSE)

---