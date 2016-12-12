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

#ifndef GLADOS_QUEUE_H_
#define GLADOS_QUEUE_H_

#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace glados
{
	template <class T>
	auto queue_limit(T t) -> std::size_t
	{
		return static_cast<std::size_t>(t);
	}

	template <class Object>
	class Queue
	{
		public:
			/*
			 * The default constructed Queue has limit 2, hence the member is 2.
			 */
			Queue() : limit_{10u}, count_{0u} {}
			explicit Queue(std::size_t limit) : limit_{limit}, count_{0u} {}

			/*
			 * Item and Object are of the same type but we need this extra template to make use of the
			 * nice reference collapsing rules
			 */
			template <class Item>
			void push(Item&& item)
			{
				auto lock = std::unique_lock<decltype(mutex_)>{mutex_};
				if(limit_ != 0u)
				{
					while(count_ >= limit_)
						count_cv_.wait(lock);
				}

				queue_.push(std::forward<Item>(item));

				if(limit_ != 0u)
					++count_;

				item_cv_.notify_one();
			}

			Object take()
			{
				auto lock = std::unique_lock<decltype(mutex_)>{mutex_};
				while(queue_.empty())
					item_cv_.wait(lock);

				auto ret = std::move(queue_.front());
				queue_.pop();

				if(limit_ != 0u)
				{
					--count_;
					count_cv_.notify_one();
				}

				return ret;
			}

		private:
			const std::size_t limit_;
			std::size_t count_;
			mutable std::mutex mutex_;
			std::condition_variable item_cv_, count_cv_;
			std::queue<Object> queue_;

	};
}

#endif /* GLADOS_QUEUE_H_ */
