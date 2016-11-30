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

#ifndef UDPSERVER_H_
#define UDPSERVER_H_

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdexcept>
#include <cstring>

namespace risa {

class udp_client_server_runtime_error : public std::runtime_error
{
public:
    udp_client_server_runtime_error(const char *w) : std::runtime_error(w) {}
};

class UDPServer
{
public:
  UDPServer(const std::string& addr, int port);
  ~UDPServer();

  int                 get_socket() const;
  int                 get_port() const;
  std::string         get_addr() const;

  int                 recv(char *msg, size_t max_size);
  int                 timed_recv(char *msg, size_t max_size, int max_wait_ms);

private:
  int                 f_socket;
  int                 f_port;
  std::string         f_addr;
  struct addrinfo *   f_addrinfo;
};

}

#endif /* UDPSERVER_H_ */
