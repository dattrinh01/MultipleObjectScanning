#include "yaml-cpp/yaml.h"
#include "yaml-cpp/node/parse.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

struct Power {
	std::string name;
	int damage;
};

struct Vec3 {
	float x, y, z;
};

void operator >> (const YAML::Node& node, Vec3& v) {
	node[0] >> v.x;
	node[1] >> v.y;
	node[2] >> v.z;
}

void operator >> (const YAML::Node& node, Power& power) {
	node["name"] >> power.name;
	node["damage"] >> power.damage;
}

int main()
{
	std::ifstream fin("monsters.yaml");
	YAML::Parser parser(fin);
	YAML::Node doc;
	parser.GetNextDocument(doc);
	for (unsigned i = 0; i < doc.size(); i++) {
		Monster monster;
		doc[i] >> monster;
		std::cout << monster.name << "\n";
	}

	return 0;
}
