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

#ifndef PIPELINE_STAGE_H_
#define PIPELINE_STAGE_H_

#include <thread>
#include <utility>

#include "../Image.h"

#include "InputSide.h"
#include "OutputSide.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class Implementation>
		class Stage
		: public InputSide<typename Implementation::input_type>
		, public OutputSide<typename Implementation::output_type>
		, public Implementation
		{
			public:
				using input_type = typename Implementation::input_type;
				using output_type = typename Implementation::output_type;

			public:
				template <typename... Args>
				Stage(Args&&... args)
				: InputSide<input_type>()
				, OutputSide<output_type>()
				, Implementation(std::forward<Args>(args)...)
				{
				}

				auto run() -> void
				{
					auto push_thread = std::thread{&Stage::push, this};
					auto take_thread = std::thread{&Stage::take, this};

					push_thread.join();
					take_thread.join();
				}

				auto push() -> void
				{
					while(true)
					{
						auto img = this->input_queue_.take();
						if(img.valid())
						   Implementation::process(std::move(img));
						else
						{
							// received poisonous pill, time to die
						   Implementation::process(std::move(img));
							break;
						}
					}
				}

				auto take() -> void
				{
					while(true)
					{
						auto result = Implementation::wait();
						if(result.valid())
							this->output(std::move(result));
						else
						{
							this->output(std::move(result));
							break;
						}
					}
				}
		};
	}
}


#endif /* PIPELINE_STAGE_H_ */
