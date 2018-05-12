/*
 * TextureMesh.cpp
 *
 * Copyright (c) 2014-2015 I23D
 *
 *
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * Additional Terms:
 *
 *      You are required to preserve legal notices and author attributions in
 *      that material or in the Appropriate Legal Notices displayed by works
 *      containing it.
 */

#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#include <boost/program_options.hpp>
#include <fstream>
#include <vector>


using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

#define APPNAME _T("TextureMeshViwo")


// S T R U C T S ///////////////////////////////////////////////////

namespace OPT {
String strInputFileName;
String strOutputFileName;
String strMeshFileName;
unsigned nResolutionLevel;
unsigned nMinResolution;
float fOutlierThreshold;
float fRatioDataSmoothness;
bool bGlobalSeamLeveling;
bool bLocalSeamLeveling;
unsigned nTextureSizeMultiple;
unsigned nRectPackingHeuristic;
uint32_t nColEmpty;
unsigned nArchiveType;
int nProcessPriority;
unsigned nMaxThreads;
String strConfigFileName;
boost::program_options::variables_map vm;

String strViwoConfigFileName;
std::vector<String> inputFileNames;
std::vector<String> transFileNames;
} // namespace OPT

// initialize and parse the command line parameters
bool Initialize(size_t argc, LPCTSTR* argv)
{
	// initialize log and console
	OPEN_LOG();
	OPEN_LOGCONSOLE();

	// group of options allowed only on command line
	boost::program_options::options_description generic("Generic options");
	generic.add_options()
		("help,h", "produce this help message")
		("working-folder,w", boost::program_options::value<std::string>(&WORKING_FOLDER), "working directory (default current directory)")
		("config-file,c", boost::program_options::value<std::string>(&OPT::strConfigFileName)->default_value(APPNAME _T(".cfg")), "file name containing program options")
		("viwo-config-file", boost::program_options::value<std::string>(&OPT::strViwoConfigFileName), "file name containing Viwo program options")
		("archive-type", boost::program_options::value<unsigned>(&OPT::nArchiveType)->default_value(2), "project archive type: 0-text, 1-binary, 2-compressed binary")
		("process-priority", boost::program_options::value<int>(&OPT::nProcessPriority)->default_value(-1), "process priority (below normal by default)")
		("max-threads", boost::program_options::value<unsigned>(&OPT::nMaxThreads)->default_value(0), "maximum number of threads (0 for using all available cores)")
		#if TD_VERBOSE != TD_VERBOSE_OFF
		("verbosity,v", boost::program_options::value<int>(&g_nVerbosityLevel)->default_value(
			#if TD_VERBOSE == TD_VERBOSE_DEBUG
			3
			#else
			2
			#endif
			), "verbosity level")
		#endif
		;

	// group of options allowed both on command line and in config file
	boost::program_options::options_description config("Refine options");
	config.add_options()
		("input-file,i", boost::program_options::value<std::string>(&OPT::strInputFileName), "input filename containing camera poses and image list")
		("output-file,o", boost::program_options::value<std::string>(&OPT::strOutputFileName), "output filename for storing the mesh")
		("resolution-level", boost::program_options::value<unsigned>(&OPT::nResolutionLevel)->default_value(1), "how many times to scale down the images before mesh refinement")
		("min-resolution", boost::program_options::value<unsigned>(&OPT::nMinResolution)->default_value(640), "do not scale images lower than this resolution")
		("outlier-threshold", boost::program_options::value<float>(&OPT::fOutlierThreshold)->default_value(6e-2f), "threshold used to find and remove outlier face textures (0 - disabled)")
		("cost-smoothness-ratio", boost::program_options::value<float>(&OPT::fRatioDataSmoothness)->default_value(0.1f), "ratio used to adjust the preference for more compact patches (1 - best quality/worst compactness, ~0 - worst quality/best compactness)")
		("global-seam-leveling", boost::program_options::value<bool>(&OPT::bGlobalSeamLeveling)->default_value(true), "generate uniform texture patches using global seam leveling")
		("local-seam-leveling", boost::program_options::value<bool>(&OPT::bLocalSeamLeveling)->default_value(true), "generate uniform texture patch borders using local seam leveling")
		("texture-size-multiple", boost::program_options::value<unsigned>(&OPT::nTextureSizeMultiple)->default_value(0), "texture size should be a multiple of this value (0 - power of two)")
		("patch-packing-heuristic", boost::program_options::value<unsigned>(&OPT::nRectPackingHeuristic)->default_value(3), "specify the heuristic used when deciding where to place a new patch (0 - best fit, 3 - good speed, 100 - best speed)")
		("empty-color", boost::program_options::value<uint32_t>(&OPT::nColEmpty)->default_value(0x00FF7F27), "color used for faces not covered by any image")
		("mesh-file", boost::program_options::value<std::string>(&OPT::strMeshFileName), "mesh file name to texture (overwrite the existing mesh)")
		;

	// hidden options, allowed both on command line and
	// in config file, but will not be shown to the user
	boost::program_options::options_description hidden("Hidden options");
/*	hidden.add_options()
		;
*/
	boost::program_options::options_description cmdline_options;
	cmdline_options.add(generic).add(config).add(hidden);

	boost::program_options::options_description config_file_options;
	config_file_options.add(config).add(hidden);

	boost::program_options::positional_options_description p;
	p.add("input-file", -1);

	try {
		// parse command line options
		boost::program_options::store(boost::program_options::command_line_parser((int)argc, argv).options(cmdline_options).positional(p).run(), OPT::vm);
		boost::program_options::notify(OPT::vm);
		INIT_WORKING_FOLDER;
		// parse configuration file
		std::ifstream ifs(MAKE_PATH_SAFE(OPT::strConfigFileName));
		if (ifs) {
			boost::program_options::store(parse_config_file(ifs, config_file_options), OPT::vm);
			boost::program_options::notify(OPT::vm);
		}
	}
	catch (const std::exception& e) {
		LOG(e.what());
		return false;
	}

	// initialize the log file
	OPEN_LOGFILE(MAKE_PATH(APPNAME _T("-")+Util::getUniqueName(0)+_T(".log")).c_str());

	// print application details: version and command line
	Util::LogBuild();
	LOG(_T("Command line:%s"), Util::CommandLineToString(argc, argv).c_str());

	/*
	// validate input
	Util::ensureValidPath(OPT::strInputFileName);
	Util::ensureUnifySlash(OPT::strInputFileName);
	Util::ensureValidPath(OPT::strOtherInputFileName);
	Util::ensureUnifySlash(OPT::strOtherInputFileName);
	if (OPT::vm.count("help") || OPT::strInputFileName.IsEmpty()) {
		boost::program_options::options_description visible("Available options");
		visible.add(generic).add(config);
		GET_LOG() << visible;
	}
	if (OPT::strInputFileName.IsEmpty())
		return false;

	// initialize optional options
	Util::ensureValidPath(OPT::strOutputFileName);
	Util::ensureUnifySlash(OPT::strOutputFileName);
	if (OPT::strOutputFileName.IsEmpty())
		OPT::strOutputFileName = Util::getFullFileName(OPT::strInputFileName) + _T(".mvs");
	*/

	if (OPT::vm.count("help") || OPT::strViwoConfigFileName.IsEmpty()) {
		boost::program_options::options_description visible("Available options");
		visible.add(generic).add(config);
		GET_LOG() << visible;
	}
	//Load Viwo Config File
	if (!OPT::strViwoConfigFileName.IsEmpty()) {
		std::ifstream inf(OPT::strViwoConfigFileName);
		inf >> OPT::strMeshFileName;
		inf >> OPT::strOutputFileName;
		Util::ensureValidPath(OPT::strMeshFileName);
		Util::ensureUnifySlash(OPT::strMeshFileName);
		Util::ensureValidPath(OPT::strOutputFileName);
		Util::ensureUnifySlash(OPT::strOutputFileName);
		VERBOSE("%s", OPT::strMeshFileName.c_str());
		VERBOSE("%s", OPT::strOutputFileName.c_str());
		int model_num = 0;
		inf >> model_num;
		VERBOSE("%d project(s) to be loaded", model_num);
		while (model_num--) {
			String inputfile_name;
			String transfile_name;
			inf >> inputfile_name;
			inf >> transfile_name;
			Util::ensureValidPath(inputfile_name);
			Util::ensureUnifySlash(inputfile_name);
			Util::ensureValidPath(transfile_name);
			Util::ensureUnifySlash(transfile_name);
			OPT::inputFileNames.push_back(inputfile_name);
			OPT::transFileNames.push_back(transfile_name);
			VERBOSE("%s", inputfile_name.c_str());
			VERBOSE("%s", transfile_name.c_str());
		}
	} else {
		return false;
	}

	// initialize global options
	Process::setCurrentProcessPriority((Process::Priority)OPT::nProcessPriority);
	#ifdef _USE_OPENMP
	if (OPT::nMaxThreads != 0)
		omp_set_num_threads(OPT::nMaxThreads);
	#endif

	#ifdef _USE_BREAKPAD
	// start memory dumper
	MiniDumper::Create(APPNAME, WORKING_FOLDER);
	#endif
	return true;
}

// finalize application instance
void Finalize()
{
	#if TD_VERBOSE != TD_VERBOSE_OFF
	// print memory statistics
	Util::LogMemoryInfo();
	#endif

	CLOSE_LOGFILE();
	CLOSE_LOGCONSOLE();
	CLOSE_LOG();
}

void TransformCoord(double *LocalCoord, float &x, float &y, float &z) {

	double newX = LocalCoord[0] * x + LocalCoord[4] * y + LocalCoord[8] * z + LocalCoord[12];
	double newY = LocalCoord[1] * x + LocalCoord[5] * y + LocalCoord[9] * z + LocalCoord[13];
	double newZ = LocalCoord[2] * x + LocalCoord[6] * y + LocalCoord[10] * z + LocalCoord[14];

	x = newX;
	y = newY;
	z = newZ;
}

bool TransMesh(String strTransMatFileName, Scene& scene) {
	FILE *fp = fopen(strTransMatFileName.c_str(), "rb");
	if (fp == NULL)
		return false;
	double transMat[16] = {0};
	fread(transMat, sizeof(transMat), 1, fp);
	FOREACH(idxVert, scene.mesh.vertices) {
		MVS::Mesh::Vertex &vert = scene.mesh.vertices[idxVert];
		TransformCoord(transMat, vert[0], vert[1], vert[2]);
	}

	fclose(fp);
	return true;
}
bool LoadTransMatFile(String strTransMatFileName, Scene& scene) {
	FILE *fp = fopen(strTransMatFileName.c_str(), "rb");
	if (fp == NULL)
		return false;
	double transMat[16] = {0};
	fread(transMat, sizeof(transMat), 1, fp);
	double s = sqrt(transMat[0]*transMat[0] + transMat[1]*transMat[1] + transMat[2]*transMat[2]);
	Point3 t(transMat[12], transMat[13], transMat[14]);
	Matrix3x3 R;
	R[0] = transMat[0]/s;
	R[1] = transMat[4]/s;
	R[2] = transMat[8]/s;
	R[3] = transMat[1]/s;
	R[4] = transMat[5]/s;
	R[5] = transMat[9]/s;
	R[6] = transMat[2]/s;
	R[7] = transMat[6]/s;
	R[8] = transMat[10]/s;
	VERBOSE("s = %lf", s);
	VERBOSE("r = \n%s", cvMat2String(R).c_str());
	VERBOSE("t = \n%s", cvMat2String(t).c_str());
	FOREACH(idxPlatform, scene.platforms) {
		Platform& platform = scene.platforms[idxPlatform];
		VERBOSE("platform:%s has %d cameras, %d poses", platform.name.c_str(), platform.cameras.GetSize(), platform.poses.GetSize());
//		FOREACH(idxCamera, platform.cameras) {
//			CameraIntern& cam = platform.cameras[idxCamera];
//			//cam.R = cam.R * R.t();
//			//cam.C = R*cam.C*s + t;
//		}
		FOREACH(idxPose, platform.poses) {
			Platform::Pose& pose = platform.poses[idxPose];
			pose.R = pose.R * R.t();
			pose.C = R*pose.C*s + t;
//			VERBOSE("PoseR%d = \n%s", idxPose, cvMat2String(pose.R).c_str());
//			VERBOSE("PoseC%d = \n%s", idxPose, cvMat2String(pose.C).c_str());
		}
	}
/*
	FOREACH(idxImage, scene.images) {
		Camera cam = scene.images[idxImage].camera;
		cam.Transform(R,t,s);
		VERBOSE("Correct P%d = \n%s", idxImage, cvMat2String(cam.P).c_str());
	}

	FOREACH(idxVert, scene.mesh.vertices) {
		MVS::Mesh::Vertex &vert = scene.mesh.vertices[idxVert];
		TransformCoord(transMat, vert[0], vert[1], vert[2]);
	}
*/
	fclose(fp);
	return true;
}

int main(int argc, LPCTSTR* argv)
{
	#ifdef _DEBUGINFO
	// set _crtBreakAlloc index to stop in <dbgheap.c> at allocation
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);// | _CRTDBG_CHECK_ALWAYS_DF);
	#endif

	if (!Initialize(argc, argv))
		return EXIT_FAILURE;

	Scene scene(OPT::nMaxThreads);

	if (!OPT::strViwoConfigFileName.IsEmpty()) {
		if (OPT::inputFileNames.empty() || OPT::transFileNames.empty()) {
			return EXIT_FAILURE;
		}

		String working_folder_full_bak = WORKING_FOLDER_FULL;
		WORKING_FOLDER_FULL = Util::getFilePath(OPT::inputFileNames[0]) + "intermediate/";
		if (!scene.Load(MAKE_PATH_SAFE(OPT::inputFileNames[0])))
			return EXIT_FAILURE;

		if (!OPT::strMeshFileName.IsEmpty()) {
			scene.mesh.Load(MAKE_PATH_SAFE(OPT::strMeshFileName));
//			TransMesh(OPT::transFileNames[0], scene);
		}

		LoadTransMatFile(OPT::transFileNames[0], scene);

		for (int i = 1; i < OPT::inputFileNames.size(); i++) {
			Scene otherScene(OPT::nMaxThreads);
			WORKING_FOLDER_FULL = Util::getFilePath(OPT::inputFileNames[i]) + "intermediate/";
			if (!otherScene.Load(MAKE_PATH_SAFE(OPT::inputFileNames[i])))
				return EXIT_FAILURE;

			LoadTransMatFile(OPT::transFileNames[i], otherScene);
			scene.Combine(otherScene);
		}
		WORKING_FOLDER_FULL = working_folder_full_bak;
	}

	FOREACH(idxImage, scene.images) {
		const Image& image = scene.images[idxImage];
		VERBOSE("Image[%d]:%s, platformID[%d], cameraID[%d], poseID[%d]", idxImage, image.name.c_str(), image.platformID, image.cameraID, image.poseID);
	}

	if (scene.mesh.IsEmpty()) {
		VERBOSE("error: empty initial mesh");
		return EXIT_FAILURE;
	}

	VERBOSE("start map texture");
	TD_TIMER_START();
	if (!scene.TextureMesh(OPT::nResolutionLevel, OPT::nMinResolution, OPT::fOutlierThreshold, OPT::fRatioDataSmoothness, OPT::bGlobalSeamLeveling, OPT::bLocalSeamLeveling, OPT::nTextureSizeMultiple, OPT::nRectPackingHeuristic, Pixel8U(OPT::nColEmpty)))
		return EXIT_FAILURE;
	VERBOSE("Mesh texturing completed: %u vertices, %u faces (%s)", scene.mesh.vertices.GetSize(), scene.mesh.faces.GetSize(), TD_TIMER_GET_FMT().c_str());

	// save the final mesh
	const String baseFileName(MAKE_PATH_SAFE(Util::getFullFileName(OPT::strOutputFileName) + _T("_texture")));
	scene.Save(baseFileName+_T(".mvs"), (ARCHIVE_TYPE)OPT::nArchiveType);
	scene.mesh.Save(baseFileName+_T(".obj"));
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (VERBOSITY_LEVEL > 2)
		scene.ExportCamerasMLP(baseFileName+_T(".mlp"), baseFileName+_T(".obj"));
	#endif

	Finalize();
	return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/
