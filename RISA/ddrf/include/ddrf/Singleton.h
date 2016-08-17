/*
 * Singleton.h
 *
 *  Created on: 01.05.2016
 *      Author: Tobias Frust (t.frust@hzdr.de)
 *
 */

#ifndef SINGLETON_H_
#define SINGLETON_H_

namespace ddrf {

template<class C>
class Singleton {
public:
	static C* instance() {
		if (!_instance)
			_instance = new C();
		return _instance;
	}
	~Singleton() {
		_instance = 0;
	}
private:
	static C* _instance;
protected:
	Singleton() {
	}
};
template<class C> C* Singleton<C>::_instance = 0;

}

#endif /* SINGLETON_H_ */
