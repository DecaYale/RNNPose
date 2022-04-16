
import numpy as np
from scipy import spatial


def fragmentation_fps(vertices, num_frags):
  """Fragmentation by the furthest point sampling algorithm.

  The fragment centers are found by iterative selection of the vertex from
  vertices that is furthest from the already selected vertices. The algorithm
  starts with the centroid of the object model which is then discarded from the
  final set of fragment centers.

  A fragment is defined by a set of points on the object model that are the
  closest to the fragment center.

  Args:
    vertices: [num_vertices, 3] ndarray with 3D vertices of the object model.
    num_frags: Number of fragments to define.

  Returns:
    [num_frags, 3] ndarray with fragment centers and [num_vertices] ndarray
    storing for each vertex the ID of the assigned fragment.
  """
  # Start with the origin of the model coordinate system.
  frag_centers = [np.array([0., 0., 0.])]

  # Calculate distances to the center from all the vertices.
  nn_index = spatial.cKDTree(frag_centers)
  nn_dists, _ = nn_index.query(vertices, k=1)
  center_inds=[]
  for _ in range(num_frags):
    # Select the furthest vertex as the next center.
    new_center_ind = np.argmax(nn_dists)
    new_center = vertices[new_center_ind]
    frag_centers.append(vertices[new_center_ind])
    center_inds.append(new_center_ind)

    # Update the distances to the nearest center.
    nn_dists[new_center_ind] = -1
    nn_dists = np.minimum(
      nn_dists, np.linalg.norm(vertices - new_center, axis=1))

  # Remove the origin.
  frag_centers.pop(0)
  frag_centers = np.array(frag_centers)

  # Assign vertices to the fragments.
  # TODO: This information can be maintained during the FPS algorithm.
  nn_index = spatial.cKDTree(frag_centers)
  _, vertex_frag_ids = nn_index.query(vertices, k=1)

#   return frag_centers, vertex_frag_ids
  return frag_centers, np.array(center_inds), vertex_frag_ids


if __name__=="__main__":
    #test
    pass
