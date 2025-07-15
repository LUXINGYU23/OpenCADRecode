## Changelog

### v1.5 (March, 2025)

:page_facing_up: [arxiv](https://arxiv.org/abs/2412.14042v2) :hugs: [model](https://huggingface.co/filapro/cad-recode-v1.5) :hugs: [dataset](https://huggingface.co/datasets/filapro/cad-recode-v1.5)

Architecture updates:
 - Remove normals: (x, y, z, n_x, n_y, n_z) -> (x, y, z)
 - Random point sampling -> farthest point sampling (from pytorch3d)
 - Points ordered by z axis -> unsorted points

Training updates:
 - 4 x H100 -> 1 x H100
 - batch size 9 -> 18 + gradient accumulation x2
 - learning rate 1e-4 -> 2e-4
 - add 0.01 noise to all points with 0.5 probability

Dataset updates:
 - Integer value range -50...+50 -> -100...+100


<table>
  <thead>
    <tr>
      <th>Version</th> <th colspan="4">DeepCAD</th> <th colspan="4">Fusion360</th> <th colspan="4">CC3D</th>
    </tr>
    <tr>
      <th></th> <th>mean CD</th> <th>med. CD</th> <th>IoU</th> <th>IR</th> <th>mean CD</th> <th>med. CD</th> <th>IoU</th> <th>IR</th> <th>mean CD</th> <th>med. CD</th> <th>IoU</th> <th>IR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>v1</td> <td>0.308</td> <td>0.168</td> <td>87.6</td> <td>0.5</td> <td>0.425</td> <td>0.159</td> <td>83.1</td> <td>0.9</td> <td>4.76</td> <td>0.761</td> <td>64.3</td> <td>3.3</td>
    </tr>
    <tr>
      <td><strong>v1.5</strong></td> <td><strong>0.298</strong></td> <td><strong>0.157</strong></td> <td><strong>92.0</strong></td> <td><strong>0.37</strong></td> 
      <td><strong>0.354</strong></td> <td><strong>0.151</strong></td> <td><strong>87.8</strong></td> <td><strong>0.5</strong></td> 
      <td><strong>0.765</strong></td> <td><strong>0.313</strong></td> <td><strong>74.2</strong></td> <td><strong>0.3</strong></td>
    </tr>
  </tbody>
</table>

### v1 (December, 2024)

[github](https://github.com/filaPro/cad-recode/releases/tag/v1.0) :page_facing_up: [arxiv](https://arxiv.org/abs/2412.14042v1) :hugs: [model](https://huggingface.co/filapro/cad-recode) :hugs: [dataset](https://huggingface.co/datasets/filapro/cad-recode)