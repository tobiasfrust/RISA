/*
 * This file is part of the GLADOS-library.
 *
 * Copyright (C) 2016 Helmholtz-Zentrum Dresden-Rossendorf
 *
 * GLADOS is free software: You can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GLADOS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with GLADOS. If not, see <http://www.gnu.org/licenses/>.
 *
 * Date: 30 November 2016
 * Authors: Tobias Frust <t.frust@hzdr.de>
 *
 */

#ifndef GLADOS_SUBJECT_H_
#define GLADOS_SUBJECT_H_

#include <list>

#include "Observer.h"

namespace glados
{
	class Subject
	{
		public:
			virtual ~Subject() = default;

			virtual auto attach(Observer*) -> void;
			virtual auto detach(Observer*) -> void;
			virtual auto notify() -> void;

		protected:
			Subject() = default;

		private:
			std::list<Observer*> observers_;
	};
}



#endif /* SUBJECT_H_ */
