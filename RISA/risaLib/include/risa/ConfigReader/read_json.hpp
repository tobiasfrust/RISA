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
 * Date: 09 January 2017
 * Authors: Tobias Frust (FWCC) <t.frust@hzdr.de>
 *
 */

#pragma once
#ifndef RISALIB_INCLUDE_RISA_CONFIGREADER_READ_JSON_HPP_
#define RISALIB_INCLUDE_RISA_CONFIGREADER_READ_JSON_HPP_

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/log/trivial.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <tuple>

namespace risa {

	class read_json {

	public:

		auto read(const std::string& path) -> boost::property_tree::ptree {
			boost::property_tree::read_json(path, root_);
			return root_;
		}


		template <typename T>
		auto get_value(const std::string& identifier) const -> T {
			const T value = root_.get<T>(identifier);
			BOOST_LOG_TRIVIAL(debug) << identifier << ": " << value;
			return value;
		}

		template <typename key_type, typename value_type>
		auto get_element_in_list(const std::string& list_identifier, const std::string& identifier, const std::pair<std::string, key_type>& key_val) const -> value_type {
			value_type ret;
			for(auto& node : root_.get_child(list_identifier)){
				if(node.second.get<key_type>(key_val.first) == key_val.second){
					ret = node.second.get<value_type>(identifier);
				}
			}
			BOOST_LOG_TRIVIAL(debug) << list_identifier << ", " << identifier << "; (" << key_val.first << ", " << key_val.second << "): " << ret;
			return ret;
		}

		template <typename T>
		auto get_list_of_elements(const std::string& list_identifier) const -> std::vector<T> {
			std::vector<T> list{};
			for(auto& node : root_.get_child(list_identifier)){
				list.push_back(node.second.get_value<T>());
			}
			return list;
		}

	private:
		boost::property_tree::ptree root_;

	};

}



#endif /* RISALIB_INCLUDE_RISA_CONFIGREADER_READ_JSON_HPP_ */
