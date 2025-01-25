/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void ProjectionFunctionCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans2D,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute relevant intermediate forward results needed in the backward.
	float3 mean = means[idx];
	double3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	double2 dL_dmean2D = { dL_dmeans2D[idx].x, dL_dmeans2D[idx].y };

	float3 P_view = transformPoint4x3(mean, view_matrix);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
	
	glm::mat3 cov = glm::transpose(W) * glm::transpose(Vrk) * W;

	double cov_0 = float(cov[0][0]);
	double cov_1 = float(cov[0][1]);
	double cov_2 = float(cov[0][2]);
	double cov_3 = float(cov[1][1]);
	double cov_4 = float(cov[1][2]);
	double cov_5 = float(cov[2][2]);

	double det_cov = cov_0*(cov_3*cov_5 - cov_4*cov_4) - cov_1*(cov_1*cov_5 - cov_2*cov_4) + cov_2*(cov_1*cov_4 - cov_2*cov_3);
	double star_cov_0 = cov_3*cov_5 - cov_4*cov_4;
	double star_cov_1 = cov_2*cov_4 - cov_1*cov_5;
	double star_cov_2 = cov_1*cov_4 - cov_2*cov_3;
	double star_cov_3 = cov_0*cov_5 - cov_2*cov_2;
	double star_cov_4 = cov_1*cov_2 - cov_0*cov_4;
	double star_cov_5 = cov_0*cov_3 - cov_1*cov_1;

	double conv_cov_0 = star_cov_0/det_cov;
	double conv_cov_1 = star_cov_1/det_cov;
	double conv_cov_2 = star_cov_2/det_cov;
	double conv_cov_3 = star_cov_3/det_cov;
	double conv_cov_4 = star_cov_4/det_cov;
	double conv_cov_5 = star_cov_5/det_cov;
    
	double k0 = conv_cov_0*P_view.x + conv_cov_1*P_view.y + conv_cov_2*P_view.z;
	double k1 = conv_cov_1*P_view.x + conv_cov_3*P_view.y + conv_cov_4*P_view.z;
	double k2 = conv_cov_2*P_view.x + conv_cov_4*P_view.y + conv_cov_5*P_view.z;

	double K = k0*P_view.x + k1*P_view.y + k2*P_view.z - 9;

	double cone_0 = k0*k0 - K*conv_cov_0;
	double cone_1 = k0*k1 - K*conv_cov_1;
	double cone_2 = k0*k2 - K*conv_cov_2;
	double cone_3 = k1*k1 - K*conv_cov_3;
	double cone_4 = k1*k2 - K*conv_cov_4;
	double cone_5 = k2*k2 - K*conv_cov_5;

	double P0 = cone_0/(h_x*h_x);
	double P1 = cone_3/(h_y*h_y);
	double P2 = 2*cone_1/(h_x*h_y);
	double P3 = -2*cone_2/h_x;
	double P4 = -2*cone_4/h_y;
	double P5 = cone_5;

	double3 xy_param = {2.0f*P1*P3 - P2*P4, 2.0f*P0*P4 - P2*P3, 4.0f*P0*P1 - P2*P2};
	double2 point_xy = {xy_param.x/xy_param.z, xy_param.y/xy_param.z};

	double N_dev = P0*point_xy.x*point_xy.x + P1*point_xy.y*point_xy.y + P2*point_xy.x*point_xy.y - P5;
	double N = 9.0f/N_dev;
    
    double3 conic_nofilter = {N*P0, 0.5f*N*P2, N*P1};
	double det_conic_nofilter = (conic_nofilter.x * conic_nofilter.z - conic_nofilter.y * conic_nofilter.y);
	double det_inv_nofilter = 1.0f / det_conic_nofilter;
	double3 cov_fil = { conic_nofilter.z * det_inv_nofilter + 0.3f, -conic_nofilter.y * det_inv_nofilter, conic_nofilter.x * det_inv_nofilter + 0.3f };
	double det_covfil = (cov_fil.x * cov_fil.z - cov_fil.y * cov_fil.y);

	// Compute gradients of 2D covariance without filter
	double dL_dcov_x, dL_dcov_y, dL_dcov_z;
	double det_covfil2inv = 1.0f / (det_covfil*det_covfil);

	dL_dcov_x = det_covfil2inv * (-cov_fil.z * cov_fil.z * dL_dconic.x + cov_fil.y * cov_fil.z * dL_dconic.y + (det_covfil - cov_fil.x * cov_fil.z) * dL_dconic.z); 
	dL_dcov_z = det_covfil2inv * (-cov_fil.x * cov_fil.x * dL_dconic.z + cov_fil.x * cov_fil.y * dL_dconic.y + (det_covfil - cov_fil.x * cov_fil.z) * dL_dconic.x);
	dL_dcov_y = det_covfil2inv * (2 * cov_fil.y * cov_fil.z * dL_dconic.x - (det_covfil + 2 * cov_fil.y * cov_fil.y) * dL_dconic.y + 2 * cov_fil.x * cov_fil.y * dL_dconic.z); 

	double dL_dconic_nofil_x, dL_dconic_nofil_y, dL_dconic_nofil_z;
	double det_nofil2inv = det_inv_nofilter*det_inv_nofilter;

	dL_dconic_nofil_x = det_nofil2inv * (-conic_nofilter.z * conic_nofilter.z * dL_dcov_x + conic_nofilter.y * conic_nofilter.z * dL_dcov_y + (det_conic_nofilter - conic_nofilter.x * conic_nofilter.z) * dL_dcov_z); 
	dL_dconic_nofil_z = det_nofil2inv * (-conic_nofilter.x * conic_nofilter.x * dL_dcov_z + conic_nofilter.x * conic_nofilter.y * dL_dcov_y + (det_conic_nofilter - conic_nofilter.x * conic_nofilter.z) * dL_dcov_x);
	dL_dconic_nofil_y = det_nofil2inv * (2 * conic_nofilter.y * conic_nofilter.z * dL_dcov_x - (det_conic_nofilter + 2 * conic_nofilter.y * conic_nofilter.y) * dL_dcov_y + 2 * conic_nofilter.x * conic_nofilter.y * dL_dcov_z);

	// Compute gradients of cone equation
	double dL_dN = (P0)*dL_dconic_nofil_x + (0.5f*P2)*dL_dconic_nofil_y + (P1)*dL_dconic_nofil_z;

	double dL_dxc = dL_dmean2D.x + (-9.0f*(2.0f*P0*point_xy.x + P2*point_xy.y)/(N_dev*N_dev))*dL_dN;
	double dL_dyc = dL_dmean2D.y + (-9.0f*(2.0f*P1*point_xy.y + P2*point_xy.x)/(N_dev*N_dev))*dL_dN;

	double dL_dP0 = (N)*dL_dconic_nofil_x + (-9.0f*point_xy.x*point_xy.x/(N_dev*N_dev))*dL_dN + (-4.0f*P1*point_xy.x/xy_param.z)*dL_dxc + ((2.0f*P4 - 4.0f*P1*point_xy.y)/xy_param.z)*dL_dyc;
	double dL_dP1 = (N)*dL_dconic_nofil_z + (-9.0f*point_xy.y*point_xy.y/(N_dev*N_dev))*dL_dN + ((2.0f*P3 - 4.0f*P0*point_xy.x)/xy_param.z)*dL_dxc + (-4.0f*P0*point_xy.y/xy_param.z)*dL_dyc;
	double dL_dP2 = (0.5f*N)*dL_dconic_nofil_y + (-9.0f*point_xy.x*point_xy.y/(N_dev*N_dev))*dL_dN + ((-P4 + 2.0f*P2*point_xy.x)/xy_param.z)*dL_dxc + ((-P3 + 2.0f*P2*point_xy.y)/xy_param.z)*dL_dyc;
	double dL_dP3 = (2.0f*P1/xy_param.z)*dL_dxc + (-P2/xy_param.z)*dL_dyc;
	double dL_dP4 = (-P2/xy_param.z)*dL_dxc + (2.0f*P0/xy_param.z)*dL_dyc;
	double dL_dP5 = (9.0f/(N_dev*N_dev))*dL_dN;

	double dL_dcone_0 = dL_dP0/(h_x*h_x);
	double dL_dcone_1 = 2*dL_dP2/(h_x*h_y);
	double dL_dcone_2 = -2*dL_dP3/h_x;
	double dL_dcone_3 = dL_dP1/(h_y*h_y);
	double dL_dcone_4 = -2*dL_dP4/h_y;
	double dL_dcone_5 = dL_dP5;

	// Compute gradients of inverse of the covariance matrix
	double dL_dK = -conv_cov_0*dL_dcone_0 - conv_cov_1*dL_dcone_1 - conv_cov_2*dL_dcone_2 - conv_cov_3*dL_dcone_3 - conv_cov_4*dL_dcone_4 - conv_cov_5*dL_dcone_5;

	double dL_dk0 = 2*k0*dL_dcone_0 + k1*dL_dcone_1 + k2*dL_dcone_2 + P_view.x*dL_dK;
	double dL_dk1 = 2*k1*dL_dcone_3 + k0*dL_dcone_1 + k2*dL_dcone_4 + P_view.y*dL_dK;
	double dL_dk2 = 2*k2*dL_dcone_5 + k0*dL_dcone_2 + k1*dL_dcone_4 + P_view.z*dL_dK;

	double dL_dconv_cov_0 = -K*dL_dcone_0 + P_view.x*dL_dk0;
	double dL_dconv_cov_1 = -K*dL_dcone_1 + P_view.y*dL_dk0 + P_view.x*dL_dk1;
	double dL_dconv_cov_2 = -K*dL_dcone_2 + P_view.z*dL_dk0 + P_view.x*dL_dk2;
	double dL_dconv_cov_3 = -K*dL_dcone_3 + P_view.y*dL_dk1;
	double dL_dconv_cov_4 = -K*dL_dcone_4 + P_view.z*dL_dk1 + P_view.y*dL_dk2;
	double dL_dconv_cov_5 = -K*dL_dcone_5 + P_view.z*dL_dk2;

	// Compute gradients of covariance matrix and mean in camera space
	float dL_dPview_x = k0*dL_dK + conv_cov_0*dL_dk0 + conv_cov_1*dL_dk1 + conv_cov_2*dL_dk2;
	float dL_dPview_y = k1*dL_dK + conv_cov_1*dL_dk0 + conv_cov_3*dL_dk1 + conv_cov_4*dL_dk2;
	float dL_dPview_z = k2*dL_dK + conv_cov_2*dL_dk0 + conv_cov_4*dL_dk1 + conv_cov_5*dL_dk2;

	double det_cov2 = 1.0f / (det_cov*det_cov);

	double dL_dcov_0 = det_cov2*((-star_cov_0*star_cov_0)*dL_dconv_cov_0 + (-star_cov_0*star_cov_1)*dL_dconv_cov_1 + (-star_cov_0*star_cov_2)*dL_dconv_cov_2 + (cov_5*det_cov - star_cov_0*star_cov_3)*dL_dconv_cov_3 + (-cov_4*det_cov - star_cov_0*star_cov_4)*dL_dconv_cov_4 + (cov_3*det_cov - star_cov_0*star_cov_5)*dL_dconv_cov_5);
	double dL_dcov_1 = det_cov2*((-2.0f*star_cov_1*star_cov_0)*dL_dconv_cov_0 + (-cov_5*det_cov - 2.0f*star_cov_1*star_cov_1)*dL_dconv_cov_1 + (cov_4*det_cov - 2.0f*star_cov_1*star_cov_2)*dL_dconv_cov_2 + (-2.0f*star_cov_1*star_cov_3)*dL_dconv_cov_3 + (cov_2*det_cov - 2.0f*star_cov_1*star_cov_4)*dL_dconv_cov_4 + (-2.0f*cov_1*det_cov - 2.0f*star_cov_1*star_cov_5)*dL_dconv_cov_5);
	double dL_dcov_2 = det_cov2*((-2.0f*star_cov_2*star_cov_0)*dL_dconv_cov_0 + (cov_4*det_cov - 2.0f*star_cov_2*star_cov_1)*dL_dconv_cov_1 + (-cov_3*det_cov - 2.0f*star_cov_2*star_cov_2)*dL_dconv_cov_2 + (-2.0f*cov_2*det_cov - 2.0f*star_cov_2*star_cov_3)*dL_dconv_cov_3 + (cov_1*det_cov - 2.0f*star_cov_2*star_cov_4)*dL_dconv_cov_4 + (-2.0f*star_cov_2*star_cov_5)*dL_dconv_cov_5);
	double dL_dcov_3 = det_cov2*((cov_5*det_cov - star_cov_3*star_cov_0)*dL_dconv_cov_0 + (-star_cov_3*star_cov_1)*dL_dconv_cov_1 + (-cov_2*det_cov - star_cov_3*star_cov_2)*dL_dconv_cov_2 + (-star_cov_3*star_cov_3)*dL_dconv_cov_3 + (-star_cov_3*star_cov_4)*dL_dconv_cov_4 + (cov_0*det_cov - star_cov_3*star_cov_5)*dL_dconv_cov_5);
	double dL_dcov_4 = det_cov2*((-2.0f*cov_4*det_cov - 2.0f*star_cov_4*star_cov_0)*dL_dconv_cov_0 + (cov_2*det_cov - 2.0f*star_cov_4*star_cov_1)*dL_dconv_cov_1 + (cov_1*det_cov - 2.0f*star_cov_4*star_cov_2)*dL_dconv_cov_2 + (-2.0f*star_cov_4*star_cov_3)*dL_dconv_cov_3 + (-cov_0*det_cov - 2.0f*star_cov_4*star_cov_4)*dL_dconv_cov_4 + (-2.0f*star_cov_4*star_cov_5)*dL_dconv_cov_5);
	double dL_dcov_5 = det_cov2*((cov_3*det_cov - star_cov_5*star_cov_0)*dL_dconv_cov_0 + (-cov_1*det_cov - star_cov_5*star_cov_1)*dL_dconv_cov_1 + (-star_cov_5*star_cov_2)*dL_dconv_cov_2 + (cov_0*det_cov - star_cov_5*star_cov_3)*dL_dconv_cov_3 + (-star_cov_5*star_cov_4)*dL_dconv_cov_4 + (-star_cov_5*star_cov_5)*dL_dconv_cov_5);
	
	// Compute gradients of covariance matrix and mean in world space
	dL_dcov[6 * idx + 0] = (W[0][0] * W[0][0] * dL_dcov_0 + W[0][0] * W[1][0] * dL_dcov_1 + W[1][0] * W[1][0] * dL_dcov_3 + W[0][0] * W[2][0] * dL_dcov_2 + W[1][0] * W[2][0] * dL_dcov_4 + W[2][0] * W[2][0] * dL_dcov_5);
	dL_dcov[6 * idx + 3] = (W[0][1] * W[0][1] * dL_dcov_0 + W[0][1] * W[1][1] * dL_dcov_1 + W[1][1] * W[1][1] * dL_dcov_3 + W[0][1] * W[2][1] * dL_dcov_2 + W[1][1] * W[2][1] * dL_dcov_4 + W[2][1] * W[2][1] * dL_dcov_5);
	dL_dcov[6 * idx + 5] = (W[0][2] * W[0][2] * dL_dcov_0 + W[0][2] * W[1][2] * dL_dcov_1 + W[1][2] * W[1][2] * dL_dcov_3 + W[0][2] * W[2][2] * dL_dcov_2 + W[1][2] * W[2][2] * dL_dcov_4 + W[2][2] * W[2][2] * dL_dcov_5);
	dL_dcov[6 * idx + 1] = 2 * W[0][0] * W[0][1] * dL_dcov_0 + (W[0][0] * W[1][1] + W[0][1] * W[1][0]) * dL_dcov_1 + 2 * W[1][0] * W[1][1] * dL_dcov_3 + (W[0][0] * W[2][1] + W[0][1] * W[2][0]) * dL_dcov_2 + (W[1][0] * W[2][1] + W[1][1] * W[2][0]) * dL_dcov_4 + 2 * W[2][0] * W[2][1] * dL_dcov_5;
	dL_dcov[6 * idx + 2] = 2 * W[0][0] * W[0][2] * dL_dcov_0 + (W[0][0] * W[1][2] + W[0][2] * W[1][0]) * dL_dcov_1 + 2 * W[1][0] * W[1][2] * dL_dcov_3 + (W[0][0] * W[2][2] + W[0][2] * W[2][0]) * dL_dcov_2 + (W[1][0] * W[2][2] + W[1][2] * W[2][0]) * dL_dcov_4 + 2 * W[2][0] * W[2][2] * dL_dcov_5;
	dL_dcov[6 * idx + 4] = 2 * W[0][2] * W[0][1] * dL_dcov_0 + (W[0][1] * W[1][2] + W[0][2] * W[1][1]) * dL_dcov_1 + 2 * W[1][1] * W[1][2] * dL_dcov_3 + (W[0][1] * W[2][2] + W[0][2] * W[2][1]) * dL_dcov_2 + (W[1][1] * W[2][2] + W[1][2] * W[2][1]) * dL_dcov_4 + 2 * W[2][1] * W[2][2] * dL_dcov_5;

	float3 dL_dmean = transformVec4x3Transpose({ dL_dPview_x, dL_dPview_y, dL_dPview_z }, view_matrix);
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const int W, int H,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);

	dL_dmean2D[idx].x *= 0.5f*W;
	dL_dmean2D[idx].y *= 0.5f*H;
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	
	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x; 
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const int W, int H,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	ProjectionFunctionCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		W, H,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors
		);
}