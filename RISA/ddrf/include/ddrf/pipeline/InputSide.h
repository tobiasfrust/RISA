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

#ifndef PIPELINE_INPUTSIDE_H_
#define PIPELINE_INPUTSIDE_H_

#include <utility>

#include "../Queue.h"

namespace ddrf
{
	namespace pipeline
	{
		template <class InputType>
		class InputSide
		{
			public:
				auto input(InputType&& in) -> void
				{
					input_queue_.push(std::forward<InputType&&>(in));
				}

			protected:
				Queue<InputType> input_queue_;
		};
	}
}


#endif /* PIPELINE_INPUTSIDE_H_ */
