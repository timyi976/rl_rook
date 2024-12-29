#include "geometry_temp.h"

const string prefix = "./";
string defaultpath = "base_v2.txt";

void ReadFile(ifstream& ifs){
    int net_num;
    string buffer;
    vector<vector<short>> readin;

    ifs >> net_num;
    ifs.ignore(numeric_limits<streamsize>::max(), '\n');
    //#1: Readin in row way
    while (getline(ifs, buffer)){
        vector<short> row;
        stringstream ss(buffer);
        short num;

        while (ss >> num) {
            row.emplace_back(num);
        }
        readin.emplace_back(row);
    }
    ifs.close();

    //#2: Represent in column way
    vector<vector<short>> table(readin[0].size(), vector<short>(readin.size(), 0));
    for(int i=0; i<readin[0].size(); ++i){
        for(int j=0; j<readin.size(); ++j){
            table[i][j] = readin[j][i];
        }
    }

    //#3: Preprocess state space
    const int x_tile_num = table.size();
    const int y_tile_num = table[0].size();
    vector<pair<pt,pt>> netlist(net_num);
    vector<vector<short>> blockage(x_tile_num, vector<short>(y_tile_num, 0));

    for(short x=0; x<table.size(); ++x){
        for(short y=0; y<table[x].size(); ++y){
            if(table[x][y]==1){
                netlist[0].first = pt(x,y); blockage[x][y] = 1;
            }
            if(table[x][y]==2){
                netlist[1].first = pt(x,y); blockage[x][y] = 1;
            }
            if(table[x][y]==4){
                netlist[2].first = pt(x,y); blockage[x][y] = 1;
            }
            if(table[x][y]==11){
                netlist[0].second = pt(x,y); blockage[x][y] = 1;
            }
            if(table[x][y]==12){
                netlist[1].second = pt(x,y); blockage[x][y] = 1;
            }
            if(table[x][y]==13){
                netlist[2].second = pt(x,y); blockage[x][y] = 1;
            }
            if(table[x][y]==14){
                blockage[x][y] = 1;
            }
        }
    }

    //#4: Astar search router
    vector<vector<pt>> solution;
    for(int i=0; i<net_num; ++i){
        pt s = netlist[i].first;
        pt t = netlist[i].second;
        blockage[t.x][t.y] = 0;
        vector<pt> path = lee_algorithm(blockage, {s}, t);
        blockage[t.x][t.y] = 1;

        if(path.empty()) cout << "error!!!!!!" << endl;
        solution.push_back(path);
    }

    int wire_length = 0;
    for(int iter=0; iter<solution.size(); ++iter){
        const auto& path = solution[iter];
        double length = (path.size() - 1);
        wire_length += length;
        cout << iter << "-th with " << length << endl;
    }
    cout << "total: " << wire_length << endl;

    for(int i=0; i<blockage.size(); ++i){
        for(int j=0; j<blockage[i].size(); ++j){
            blockage[i][j] *= 14;
        }
    }

    //#5: visualize output
    blockage[netlist[0].first.x][netlist[0].first.y] = 1;
    blockage[netlist[1].first.x][netlist[1].first.y] = 2;
    blockage[netlist[2].first.x][netlist[2].first.y] = 4;
    blockage[netlist[0].second.x][netlist[0].second.y] = 1;
    blockage[netlist[1].second.x][netlist[1].second.y] = 2;
    blockage[netlist[2].second.x][netlist[2].second.y] = 4;
    for(int i=0; i<solution[0].size(); ++i){
        blockage[solution[0][i].x][solution[0][i].y] = 11;
    }
    for(int i=0; i<solution[1].size(); ++i){
        blockage[solution[1][i].x][solution[1][i].y] = 22;
    }
    for(int i=0; i<solution[2].size(); ++i){
        blockage[solution[2][i].x][solution[2][i].y] = 44;
    }

    for(int i=0; i<blockage[0].size(); ++i){
        for(int j=0; j<blockage.size(); ++j){
            cout << blockage[j][i] << (blockage[j][i]%10>0?"":" ") << " ";
        }
        cout << endl;
    }
}

int main(int argc, const char * argv[]){
    ifstream ifs;
    if(argc > 1) defaultpath = argv[1];
    ifs.open(prefix + defaultpath);
    if(!ifs.is_open()){
        cout << "Failed to open file.\n";
        return 1;
    }
    ReadFile(ifs);
    return 0;
}
