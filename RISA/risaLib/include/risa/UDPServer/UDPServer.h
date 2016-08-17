/*
 * http://linux.m2osw.com/c-implementation-udp-clientserver
 *
 * UDPServer.h
 *
 *  Created on: 29.06.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
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
