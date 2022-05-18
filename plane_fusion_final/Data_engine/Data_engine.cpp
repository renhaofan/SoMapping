#include "Data_engine.h"


Data_engine::Data_engine()
{
}
Data_engine::~Data_engine()
{
}


//
void Data_engine::init(Image_loader * image_loader_ptr, string output_folder, bool _is_ICL_NUIM_dataset)
{
	// Get pointer of Image_loader
	this->Data_loader::init_image_loader(image_loader_ptr);

	// Initiate Data_writer
	this->Data_writer::init(output_folder, _is_ICL_NUIM_dataset);

}

