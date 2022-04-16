
#include "neighbors.h"


void brute_neighbors(vector<PointXYZ>& queries, vector<PointXYZ>& supports, vector<int>& neighbors_indices, float radius, int verbose)
{

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	vector<vector<int>> tmp(queries.size());

	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
		int i = 0;
		for (auto& p : supports)
		{
			if ((p0 - p).sq_norm() < r2)
			{
				tmp[i0].push_back(i);
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = -1;
		}
		i0++;
	}

	return;
}

void ordered_neighbors(vector<PointXYZ>& queries,
                        vector<PointXYZ>& supports,
                        vector<int>& neighbors_indices,
                        float radius)
{

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());

	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
		int i = 0;
		for (auto& p : supports)
		{
		    d2 = (p0 - p).sq_norm();
			if (d2 < r2)
			{
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    int index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, i);

			    // Update max count
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = -1;
		}
		i0++;
	}

	return;
}

void batch_ordered_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{

	// Initialize variables
	// ******************

	// square radius
	float r2 = radius * radius;

	// indices
	int i0 = 0;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<int>> tmp(queries.size());
	vector<vector<float>> dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;


	// Search neigbors indices
	// ***********************

	for (auto& p0 : queries)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

	    // Loop only over the supports of current batch
	    vector<PointXYZ>::iterator p_it;
		int i = 0;
        for(p_it = supports.begin() + sum_sb; p_it < supports.begin() + sum_sb + s_batches[b]; p_it++ )
        {
		    d2 = (p0 - *p_it).sq_norm();
			if (d2 < r2)
			{
			    // Find order of the new point
			    auto it = std::upper_bound(dists[i0].begin(), dists[i0].end(), d2);
			    int index = std::distance(dists[i0].begin(), it);

			    // Insert element
                dists[i0].insert(it, d2);
                tmp[i0].insert(tmp[i0].begin() + index, sum_sb + i);

			    // Update max count
				if (tmp[i0].size() > max_count)
					max_count = tmp[i0].size();
			}
			i++;
		}
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	for (auto& inds : tmp)
	{
		for (int j = 0; j < max_count; j++)
		{
			if (j < inds.size())
				neighbors_indices[i0 * max_count + j] = inds[j];
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	return;
}


void batch_nanoflann_neighbors(vector<PointXYZ>& queries,
                                vector<PointXYZ>& supports,
                                vector<int>& q_batches,
                                vector<int>& s_batches,
                                vector<int>& neighbors_indices,
                                float radius)
{

	// Initialize variables
	// ******************

	// indices
	int i0 = 0;

	// Square radius
	float r2 = radius * radius;

	// Counting vector
	int max_count = 0;
	float d2;
	vector<vector<pair<size_t, float>>> all_inds_dists(queries.size());

	// batch index
	int b = 0;
	int sum_qb = 0;
	int sum_sb = 0;

	// Nanoflann related variables
	// ***************************

	// CLoud variable
	PointCloud current_cloud;

	// Tree parameters
	nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10 /* max leaf */);

	// KDTree type definition
    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud > ,
                                                        PointCloud,
                                                        3 > my_kd_tree_t;

    // Pointer to trees
    my_kd_tree_t* index;

    // Build KDTree for the first batch element
    current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);
    index = new my_kd_tree_t(3, current_cloud, tree_params);
    index->buildIndex();


	// Search neigbors indices
	// ***********************

    // Search params
    nanoflann::SearchParams search_params;
    search_params.sorted = true;

	for (auto& p0 : queries)
	{

	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;

	        // Change the points
	        current_cloud.pts.clear();
            current_cloud.pts = vector<PointXYZ>(supports.begin() + sum_sb, supports.begin() + sum_sb + s_batches[b]);

	        // Build KDTree of the current element of the batch
            delete index;
            index = new my_kd_tree_t(3, current_cloud, tree_params);
            index->buildIndex();
	    }

	    // Initial guess of neighbors size
        all_inds_dists[i0].reserve(max_count);

	    // Find neighbors
	    float query_pt[3] = { p0.x, p0.y, p0.z};
		size_t nMatches = index->radiusSearch(query_pt, r2, all_inds_dists[i0], search_params);

        // Update max count
        if (nMatches > max_count)
            max_count = nMatches;

        // Increment query idx
		i0++;
	}

	// Reserve the memory
	neighbors_indices.resize(queries.size() * max_count);
	i0 = 0;
	sum_sb = 0;
	sum_qb = 0;
	b = 0;
	for (auto& inds_dists : all_inds_dists)
	{
	    // Check if we changed batch
	    if (i0 == sum_qb + q_batches[b])
	    {
	        sum_qb += q_batches[b];
	        sum_sb += s_batches[b];
	        b++;
	    }

		for (int j = 0; j < max_count; j++)
		{
			if (j < inds_dists.size())
				neighbors_indices[i0 * max_count + j] = inds_dists[j].first + sum_sb;
			else
				neighbors_indices[i0 * max_count + j] = supports.size();
		}
		i0++;
	}

	delete index;

	return;
}

