//
//  Created by skype324 skype324 on 2023/4/4.
//
#ifndef geometry_temp_h
#define geometry_temp_h

#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <iomanip>

using namespace std;

// Define point types for easier handling of coordinates
using pt = pair<short,short>;
using Dpt = pair<double,double>;

// Simplify access to pair elements
#define x first
#define y second

// Define possible movement directions
const vector<pt> dir = {pt(1, 1), pt(2,0), pt(1, -1), pt(-1, 1), pt(-2,0), pt(-1, -1)};
const vector<vector<short>> passby = {{0,1,3},{0,1,2},{1,2,5},{0,3,4},{3,4,5},{2,4,5}};

// Overload arithmetic operators for pair manipulation, enhancing code readability and ease of geometric computations
template <typename T,typename U>
std::pair<T,U> operator+(const std::pair<T,U> & l,const std::pair<T,U> & r) {
  return {l.first + r.first, l.second + r.second};
}

template <typename T,typename U>
std::pair<T,U> operator-(const std::pair<T,U> & l,const std::pair<T,U> & r) {
  return {l.first - r.first, l.second - r.second};
}

template <typename T,typename U, typename R>
std::pair<T,U> operator*(const std::pair<T,U> & l,const R & r) {
  return {l.first * r, l.second * r};
}

template <typename T,typename U, typename R>
std::pair<T,U> operator/(const std::pair<T,U> & l,const R & r) {
  return {l.first / r, l.second / r};
}

// Stream insertion and extraction operators for pairs, facilitating input/output operations
template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p) {
  os << p.first << " , " << p.second;
  return os;
}

template<typename T, typename U>
std::istream& operator>>(std::istream& os, std::pair<T, U>& p) {
  os >> p.first >> p.second;
  return os;
}

// Equality operators for Dpt and pairs of Dpt, utilizing a tolerance level for floating point comparisons
inline bool operator==(const Dpt& l, const Dpt& r) {
  return abs(l.x - r.x) + abs(l.y - r.y) < 1e-4;
};

inline bool operator==(const pair<Dpt,Dpt>& l, const pair<Dpt,Dpt>& r) {
  return (l.first == r.first) && (l.second == r.second);
};

// Utility functions for geometric computations
double length(const Dpt& a){ return sqrt(a.x * a.x + a.y * a.y); }
double length(const pt& a){  return sqrt(a.x * a.x + a.y * a.y); }
double T1Norm(const Dpt& a){
  double p = abs(a.x);
  double q = abs(a.y);
  return max(p + q, 2 * q);
}
double T1Norm(const pt& a){
  double p = abs(a.x);
  double q = abs(a.y);
  return max(p + q, 2 * q);
}
//*******************************************************//
//  Part[4/4]: lee's algorithm                           //
//*******************************************************//
vector<pt> lee_algorithm(const vector<vector<short>>& mat, vector<pt> start_pts, pt end_pt) {

  int n = (int)mat.size(); // Number of rows in the matrix
  int m = (int)mat[0].size(); // Number of columns in the matrix
  vector<vector<short>> dist(n, vector<short>(m, -1)); // Stores distance from the start point
  vector<vector<bool>> visited(n, vector<bool>(m, false)); // Tracks visited points
  vector<vector<short>> preD(n, vector<short>(m, -1)); // Tracks previous direction for path construction

  // Initialize start points in the data structures
  for (const auto& start_pt : start_pts) {
    dist[start_pt.x][start_pt.y] = 0;
    visited[start_pt.x][start_pt.y] = true;
    preD[start_pt.x][start_pt.y] = -2;
  }

  // Custom comparator for priority queue based on T1Norm or length and additional path losses
  auto cmp = [&](const pt& a, const pt& b) {
    const double da = T1Norm(Dpt((a.x - end_pt.x), (a.y - end_pt.y)));
    const double db = T1Norm(Dpt((b.x - end_pt.x), (b.y - end_pt.y)));
    return da > db; // Priority is given to shorter or less costly paths
  };
  priority_queue<pt, vector<pt>, decltype(cmp)> q(cmp);

  // Enqueue start points
  for (const auto& start_pt : start_pts) {
    q.push(start_pt);
  }

  // Breadth-first search (BFS) with priority queue to find the shortest path
  while (!q.empty()) {
    pt curr = q.top(); q.pop();

    bool result = 0;
    // Explore all possible directions
    for (int i = 0; i < dir.size(); ++i) {
      if(preD[curr.x][curr.y]>-1 && i!=passby[preD[curr.x][curr.y]][0] && i!=passby[preD[curr.x][curr.y]][1] && i!=passby[preD[curr.x][curr.y]][2]) continue;

      int x = curr.x + dir[i].x;
      int y = curr.y + dir[i].y;


      // Check for valid movement
      if ((x == end_pt.x && y == end_pt.y) || (x >= 0 && x < n && y >= 0 && y < m && mat[x][y] != 1 && !visited[x][y])) {
        dist[x][y] = dist[curr.x][curr.y] + 1;
        preD[x][y] = i;
        visited[x][y] = true;
        q.push({x, y});

        if (x == end_pt.x && y == end_pt.y) {
          curr = {x, y}; result = 1;
          break; // Found the end point
        }
      }
    }
    if(result) break;
  }

  // Construct the path from end to start by backtracking
  vector<pt> path;
  if (dist[end_pt.x][end_pt.y] != -1) {
    pt cur = end_pt;
    while (dist[cur.x][cur.y] > 0 && preD[cur.x][cur.y] >= 0) {
      path.emplace_back(cur);
      cur = cur - dir[preD[cur.x][cur.y]];
    }
    path.emplace_back(cur); // Add the start point
    reverse(path.begin(), path.end()); // Reverse to get the correct order
  }
  return path;
}
#endif /* geometry_temp_h */
