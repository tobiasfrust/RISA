/*
 * This file is part of the RISA-library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * RISA is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RISA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with RISA. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 30 November 2016
 * Authors: Tobias Frust <t.frust@hzdr.de>
 *
 */

#ifndef PIPELINE_SINKSTAGE_H_
#define PIPELINE_SINKSTAGE_H_

#include <cstdint>
#include <string>
#include <utility>

#include <boost/log/trivial.hpp>

#include "../Filesystem.h"
#include "../Volume.h"

#include "InputSide.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class ImageSaver>
		class SinkStage : public ImageSaver
						, public InputSide<ddrf::Image<typename ImageSaver::manager_type>>
		{
			public:
				using input_type = ddrf::Image<typename ImageSaver::manager_type>;

			public:
				SinkStage(const std::string& path, const std::string& prefix, const std::string& config)
				: ImageSaver(config), InputSide<input_type>(), path_{path}, prefix_{prefix}
				{
					bool created = createDirectory(path_);
					if(!created)
						BOOST_LOG_TRIVIAL(fatal) << "SinkStage: Could not create target directory at " << path;

					if(path_.back() != '/')
						path_.append("/");
				}

				auto run() -> void
				{
					auto counter = 0;
					while(true)
					{
						auto img = this->input_queue_.take();
						if(img.valid())
						{
							auto path = path_ + prefix_ + std::to_string(0);
							BOOST_LOG_TRIVIAL(debug) << "SinkStage: Saving to " << path;
							ImageSaver::saveImage(std::move(img), path);
							++counter;
						}
						else
						{
							BOOST_LOG_TRIVIAL(info) << "SinkStage: Poisonous pill arrived, terminating.";
							break; // poisonous pill
						}
					}
				}

			private:
				std::string path_;
				std::string prefix_;
		};
	}
}


#endif /* PIPELINE_SINKSTAGE_H_ */
