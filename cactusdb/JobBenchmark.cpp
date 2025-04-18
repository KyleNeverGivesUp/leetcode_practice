#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <fcntl.h>
#include <folly/init/Init.h>
#include <json/json.h>
#include <stdlib.h>
#include <time.h>
#include <torch/torch.h>
#include <unistd.h>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "velox/common/file/FileSystems.h"
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/dwio/dwrf/reader/DwrfReader.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/ml_functions/Concat.h"
#include "velox/ml_functions/DecisionTree.h"
#include "velox/ml_functions/NNBuilder.h"
#include "velox/ml_functions/VeloxDecisionTree.h"
#include "velox/ml_functions/XGBoost.h"
#include "velox/ml_functions/functions.h"
#include "velox/ml_functions/tests/MLTestUtility.h"
#include "velox/parse/TypeResolver.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

using namespace std;
using namespace ml;
using namespace facebook::velox;
using namespace facebook::velox::test;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::core;

/*
 * The structure to describe the context of a neural network model architecture
 */
struct NNModelContext {
  int inputFeatures;

  int numLayers;

  int hiddenLayerNeurons;

  int outputLayerNeurons;
};

/*
 * The structure to describe the context of a decision tree model architecture
 */

struct DTModelContext {
  int inputFeatures;

  int treeDepth;
};

class JobBenchmark : HiveConnectorTestBase {
 public:
  JobBenchmark() {
    // Register Presto scalar functions.
    functions::prestosql::registerAllScalarFunctions();
    // Register Presto aggregate functions.
    aggregate::prestosql::registerAllAggregateFunctions();
    // Register type resolver with DuckDB SQL parser.
    parse::registerTypeResolver();
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();

    auto hiveConnector =
        connector::getConnectorFactory(
            connector::hive::HiveConnectorFactory::kHiveConnectorName)
            ->newConnector(
                kHiveConnectorId, std::make_shared<core::MemConfig>());
    connector::registerConnector(hiveConnector);

    // SetUp();
  }

  ~JobBenchmark() {}

  void SetUp() override {
    // TODO: not used for now
    // HiveConnectorTestBase::SetUp();
    // parquet::registerParquetReaderFactory();
  }

  void TearDown() override {
    HiveConnectorTestBase::TearDown();
  }

  void TestBody() override {}

  static void waitForFinishedDrivers(const std::shared_ptr<exec::Task>& task) {
    while (!task->isFinished()) {
      usleep(1000); // 0.01 second.
    }
  }

  std::shared_ptr<folly::Executor> executor_{
      std::make_shared<folly::CPUThreadPoolExecutor>(
          std::thread::hardware_concurrency())};

  std::shared_ptr<core::QueryCtx> queryCtx_{
      std::make_shared<core::QueryCtx>(executor_.get())};

  std::shared_ptr<memory::MemoryPool> pool_{
      memory::MemoryManager::getInstance()->addLeafPool()};

  VectorMaker maker{pool_.get()};

  std::unordered_map<std::string, RowVectorPtr> tableName2RowVector;
  std::unordered_map<std::string, std::vector<std::string>> tabel2Columns;

  // Function to check if a string represents a valid integer
  bool isInteger(const std::string& s) {
    if (s.empty() || (s.size() > 1 && s[0] == '0')) {
      return false; // prevent leading zeros for non-zero integers
    }

    for (char c : s) {
      if (!std::isdigit(c)) {
        return false;
      }
    }

    return true;
  }

  RowVectorPtr getTableFromCSVFile(
      VectorMaker& maker,
      std::string csvFilePath,
      std::string tableName,
      int k) {
    std::ifstream file(csvFilePath.c_str());
    if (file.fail()) {
      std::cerr << "Error in reading data file:" << csvFilePath << std::endl;
      exit(1);
    }

    std::cout << tableName << std::endl;

    std::unordered_map<std::string, std::vector<string>> colName2colHeader;

    std::unordered_map<std::string, std::vector<int>> colName2colType;

    colName2colType["aka_name"] = {0, 0, 1, 1, 1, 1, 1, 1};
    colName2colType["aka_title"] = {0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1};
    colName2colType["cast_info"] = {0, 0, 0, 0, 1, 0, 0};
    colName2colType["char_name"] = {0, 1, 1, 0, 1, 1, 1};
    colName2colType["comp_cast_type"] = {0, 1};
    colName2colType["company_name"] = {0, 1, 1, 0, 1, 1, 1};
    colName2colType["company_type"] = {0, 1};
    colName2colType["complete_cast"] = {0, 0, 0, 0};
    colName2colType["info_type"] = {0, 1};
    colName2colType["keyword"] = {0, 1, 1};
    colName2colType["kind_type"] = {0, 1};
    colName2colType["link_type"] = {0, 1};
    colName2colType["movie_companies"] = {0, 0, 0, 0, 1};
    colName2colType["movie_info_idx"] = {0, 0, 0, 1, 1};
    colName2colType["movie_keyword"] = {0, 0, 0};
    colName2colType["movie_link"] = {0, 0, 0, 0};
    colName2colType["name"] = {0, 1, 1, 0, 1, 1, 1, 1, 1};
    colName2colType["role_type"] = {0, 1};
    colName2colType["title"] = {0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1};
    colName2colType["movie_info"] = {0, 0, 0, 1, 1};
    colName2colType["person_info"] = {0, 0, 0, 1, 1};

    std::vector<std::string> aka_name_columns = {
        "id",
        "person_id",
        "name",
        "imdb_index",
        "name_pcode_cf",
        "name_pcode_ndf",
        "surname_pcode",
        "md5sum",
        "an_features"};

    colName2colHeader["aka_name"] = aka_name_columns;

    std::vector<std::string> aka_title_columns = {
        "id",
        "movie_id",
        "title",
        "imdb_index",
        "kind_id",
        "production_year",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "note",
        "md5sum",
        "at_features"};

    colName2colHeader["aka_title"] = aka_title_columns;

    std::vector<std::string> cast_info_columns = {
        "id",
        "person_id",
        "movie_id",
        "person_role_id",
        "note",
        "nr_order",
        "role_id",
        "ci_features"};

    colName2colHeader["cast_info"] = cast_info_columns;

    std::vector<std::string> char_name_columns = {
        "id",
        "name",
        "imdb_index",
        "imdb_id",
        "name_pcode_nf",
        "surname_pcode",
        "md5sum",
        "chn_features"};

    colName2colHeader["char_name"] = char_name_columns;

    std::vector<std::string> comp_cast_type_columns = {
        "id", "kind", "cct_features"};

    colName2colHeader["comp_cast_type"] = comp_cast_type_columns;

    std::vector<std::string> company_name_columns = {
        "id",
        "name",
        "country_code",
        "imdb_id",
        "name_pcode_nf",
        "name_pcode_sf",
        "md5sum",
        "cn_features"};

    colName2colHeader["company_name"] = company_name_columns;

    std::vector<std::string> company_type_columns = {
        "id", "kind", "ct_features"};

    colName2colHeader["company_type"] = company_type_columns;

    std::vector<std::string> complete_cast_columns = {
        "id", "movie_id", "subject_id", "status_id", "cc_features"};

    colName2colHeader["complete_cast"] = complete_cast_columns;

    std::vector<std::string> info_type_columns = {"id", "info", "it_features"};

    colName2colHeader["info_type"] = info_type_columns;

    std::vector<std::string> keyword_columns = {
        "id", "keyword", "phonetic_code", "k_features"};

    colName2colHeader["keyword"] = keyword_columns;

    std::vector<std::string> kind_type_columns = {"id", "kind", "kt_features"};

    colName2colHeader["kind_type"] = kind_type_columns;

    std::vector<std::string> link_type_columns = {"id", "link", "lt_features"};

    colName2colHeader["link_type"] = link_type_columns;

    std::vector<std::string> movie_companies_columns = {
        "id",
        "movie_id",
        "company_id",
        "company_type_id",
        "note",
        "mc_features"};

    colName2colHeader["movie_companies"] = movie_companies_columns;

    std::vector<std::string> movie_info_idx_columns = {
        "id", "movie_id", "info_type_id", "info", "note", "mii_features"};

    colName2colHeader["movie_info_idx"] = movie_info_idx_columns;

    std::vector<std::string> movie_keyword_columns = {
        "id", "movie_id", "keyword_id", "mk_features"};

    colName2colHeader["movie_keyword"] = movie_keyword_columns;

    std::vector<std::string> movie_link_columns = {
        "id", "movie_id", "linked_movie_id", "link_type_id", "ml_features"};

    colName2colHeader["movie_link"] = movie_link_columns;

    std::vector<std::string> name_columns = {
        "id",
        "name",
        "imdb_index",
        "imdb_id",
        "gender",
        "name_pcode_cf",
        "name_pcode_nf",
        "surname_pcode",
        "md5sum",
        "n_features"};

    colName2colHeader["name"] = name_columns;

    std::vector<std::string> role_type_columns = {"id", "role", "rt_features"};

    colName2colHeader["role_type"] = role_type_columns;

    std::vector<std::string> title_columns = {
        "id",
        "title",
        "imdb_index",
        "kind_id",
        "production_year",
        "imdb_id",
        "phonetic_code",
        "episode_of_id",
        "season_nr",
        "episode_nr",
        "series_years",
        "md5sum",
        "t_features"};

    colName2colHeader["title"] = title_columns;

    std::vector<std::string> movie_info_columns = {
        "id", "movie_id", "info_type_id", "info", "note", "mi_features"};

    colName2colHeader["movie_info"] = movie_info_columns;

    std::vector<std::string> person_info_columns = {
        "id", "person_id", "info_type_id", "info", "note", "pi_features"};

    colName2colHeader["person_info"] = person_info_columns;

    std::string line;

    std::vector<std::vector<int>> intCols;

    std::vector<std::vector<std::string>> stringCols;

    std::vector<int> colTypeIndex = colName2colType[tableName];

    std::vector<int> colIndexInType;

    int colIndex = 0;

    std::string cell;

    int numRows = 0;

    while (std::getline(file, line)) {
      // std::cout << line << std::endl;

      // analyze the first line
      std::stringstream iss(line);

      bool fragmentFlag = false;

      std::string fragmentedStr;

      colIndex = 0;

      // The JOB tables only have two types of columns: integer and string
      while (std::getline(iss, cell, ',')) {
        if ((fragmentFlag == false) && (cell.size() == 1) && (cell[0] == '"')) {
          fragmentFlag = true;

          fragmentedStr = ",";

          continue;

        } else if (
            (fragmentFlag == true) && (cell.size() == 1) && (cell[0] == '"')) {
          fragmentFlag = false;

          cell = fragmentedStr;

          fragmentedStr = "";

        } else if (
            (fragmentFlag == false) && (cell[0] == '"') &&
            ((cell[cell.size() - 1] != '"') ||
             ((cell[cell.size() - 1] == '"') &&
              (cell[cell.size() - 2] == '\\')))) {
          fragmentFlag = true;

          fragmentedStr = cell;

          continue;

        } else if (
            (fragmentFlag == true) && (cell[0] != '"') &&
            (cell[cell.size() - 1] == '"') && (cell[cell.size() - 2] != '\\')) {
          fragmentFlag = false;

          fragmentedStr += cell;

          cell = fragmentedStr;

          fragmentedStr = "";

        } else if (fragmentFlag == true) {
          fragmentedStr += cell;

          continue;
        }

        // std::cout << colIndex << ":" << cell << std::endl;

        if (!fragmentFlag) {
          if (!colTypeIndex[colIndex]) {
            // this is an integer column

            if (numRows == 0) {
              if (cell == "")

                intCols.push_back(std::vector<int>{INT_MIN});

              else

                intCols.push_back(std::vector<int>{stoi(cell)});

              colIndexInType.push_back(intCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[colIndex];

              if (cell == "")

                intCols[vecIndex].push_back(INT_MIN);

              else

                intCols[vecIndex].push_back(stoi(cell));
            }

          } else {
            // this is a string column

            if (numRows == 0) {
              stringCols.push_back(std::vector<std::string>{cell});

              colIndexInType.push_back(stringCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[colIndex];

              stringCols[vecIndex].push_back(cell);
            }
          }

          colIndex++;
        }
      }

      if (colIndex < colTypeIndex.size()) {
        // std::cout << "colIndex:" << colIndex << std::endl;
        // std::cout << "colTypeIndex.size():"<< colTypeIndex.size() <<
        // std::endl;

        for (int i = colIndex; i < colTypeIndex.size(); i++) {
          if (!colTypeIndex[i]) {
            if (numRows == 0) {
              intCols.push_back(std::vector<int>{INT_MIN});

              colIndexInType.push_back(intCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[i];

              intCols[vecIndex].push_back(INT_MIN);
            }

          } else {
            if (numRows == 0) {
              stringCols.push_back(std::vector<std::string>{""});

              colIndexInType.push_back(stringCols.size() - 1);

            } else {
              int vecIndex = colIndexInType[i];

              stringCols[vecIndex].push_back("");
            }
          }
        }
      }

      colIndex = colTypeIndex.size();

      /*if (numRows == 0) {

          for (int i = 0; i < colIndex; i++) {

               std::cout << colTypeIndex[i] << ":" << colIndexInType[i] <<
      std::endl;

          }

      }*/

      numRows++;
    }

    std::vector<VectorPtr> vecs;

    std::cout << "Building RowVector for this table with " << colIndex
              << " columns and " << numRows << " rows." << std::endl;

    for (int i = 0; i < colIndex; i++) {
      int type = colTypeIndex[i];

      int vecIndex = colIndexInType[i];

      // std::cout << i << ":" << type << ":" << vecIndex << std::endl;

      if (!type) {
        auto vec = maker.flatVector<int>(intCols[vecIndex]);

        vecs.push_back(vec);

      } else {
        auto vec = maker.flatVector<std::string>(stringCols[vecIndex]);

        vecs.push_back(vec);
      }
    }

    // to create the last column, which is a feature vector of length k

    if (k < 0)
      k = 8;

    std::vector<std::vector<float>> inputVectors;

    for (int i = 0; i < numRows; i++) {
      std::vector<float> inputVector;

      for (int j = 0; j < k; j++) {
        if (j % 2 == 0)

          inputVector.push_back(1.0);

        else

          inputVector.push_back(0.0);
      }

      inputVectors.push_back(inputVector);
    }

    auto inputArrayVector = maker.arrayVector<float>(inputVectors, REAL());

    vecs.push_back(inputArrayVector);

    RowVectorPtr myRowVector =
        maker.rowVector(colName2colHeader[tableName], vecs);

    return myRowVector;
  }

  int sampleQuery() {
    return 29;
  }

  int sampleModel() {
    return 0;
  }

  void sampleNNModelArch(int numInputFeatures, NNModelContext& nn) {
    nn.inputFeatures = numInputFeatures;
    nn.numLayers = 3;
    nn.hiddenLayerNeurons = 16;
    nn.outputLayerNeurons = 2;
  }

  void sampleDTModelArch(int numInputFeatures, DTModelContext& dt) {
    dt.inputFeatures = numInputFeatures;
    dt.treeDepth = 8;
  }

  bool
  replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
      return false;
    str.replace(start_pos, from.length(), to);
    return true;
  }

  /*
   * SELECT MIN(chn.name) AS voiced_char,
       MIN(n.name) AS voicing_actress,
       MIN(t.title) AS voiced_animation
  FROM aka_name AS an,
       complete_cast AS cc,
       comp_cast_type AS cct1,
       comp_cast_type AS cct2,
       char_name AS chn,
       cast_info AS ci,
       company_name AS cn,
       info_type AS it,
       info_type AS it3,
       keyword AS k,
       movie_companies AS mc,
       movie_info AS mi,
       movie_keyword AS mk,
       name AS n,
       person_info AS pi,
       role_type AS rt,
       title AS t
  WHERE cct1.kind ='cast'
    AND cct2.kind ='complete+verified'
    AND chn.name = 'Queen'
    AND ci.note IN ('(voice)',
                    '(voice) (uncredited)',
                    '(voice: English version)')
    AND cn.country_code ='[us]'
   aaz AND it.info = 'release dates'
    AND it3.info = 'trivia'
    AND k.keyword = 'computer-animation'
    AND mi.info IS NOT NULL
    AND (mi.info LIKE 'Japan:%200%'
         OR mi.info LIKE 'USA:%200%')
    AND n.gender ='f'
    AND n.name LIKE '%An%'
    AND rt.role ='actress'
    AND t.title = 'Shrek 2'
    AND t.production_year BETWEEN 2000 AND 2010
    AND t.id = mi.movie_id
    AND t.id = mc.movie_id
    AND t.id = ci.movie_id
    AND t.id = mk.movie_id
    AND t.id = cc.movie_id
    AND mc.movie_id = ci.movie_id
    AND mc.movie_id = mi.movie_id
    AND mc.movie_id = mk.movie_id
    AND mc.movie_id = cc.movie_id
    AND mi.movie_id = ci.movie_id
    AND mi.movie_id = mk.movie_id
    AND mi.movie_id = cc.movie_id
    AND ci.movie_id = mk.movie_id
    AND ci.movie_id = cc.movie_id
    AND mk.movie_id = cc.movie_id
    AND cn.id = mc.company_id
    AND it.id = mi.info_type_id
    AND n.id = ci.person_id
    AND rt.id = ci.role_id
    AND n.id = an.person_id
    AND ci.person_id = an.person_id
    AND chn.id = ci.person_role_id
    AND n.id = pi.person_id
    AND ci.person_id = pi.person_id
    AND it3.id = pi.info_type_id
    AND k.id = mk.keyword_id
    AND cct1.id = cc.subject_id
    AND cct2.id = cc.status_id;
    */

    std::unordered_map<std::string, PlanBuilder> getBaseTableSourcesQ29(std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
      std::unordered_map<std::string, PlanBuilder>
        sources; // with filters and projections pushed down;

    auto an_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["aka_name"]})
                    .project({"person_id as an_person_id", "an_features"});
    //  .capturePlanNodeId(akaNameNodeId1)
    tabel2Columns["an"] = {"an_person_id", "an_features"};

    sources["an"] = an_a;

    auto cc_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["complete_cast"]})
            .project({"movie_id", "subject_id", "status_id", "cc_features"});
    //  .capturePlanNodeId(completeCastNodeId1)
    tabel2Columns["cc"] = {"movie_id", "subject_id", "status_id", "cc_features"};

    sources["cc"] = cc_a;

    auto cct1_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .filter("kind = 'cast'")
            .project({"id as cct1_id", "cct_features as cct1_features"});
    //  .capturePlanNodeId(completeCastNodeId1)
    tabel2Columns["cct1"] = {"cct1_id", "cct1_features"};

    sources["cct1"] = cct1_a;

    auto cct2_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .filter("kind ='complete+verified'")
            .project({"id as cct2_id", "cct_features as cct2_features"});
    //  .capturePlanNodeId(compCastTypeNodeId2)
    tabel2Columns["cct2"] = {"cct2_id", "cct2_features"};

    sources["cct2"] = cct2_a;

    auto chn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["char_name"]})
                     .filter("name = 'Queen'")
                     .project({"id as chn_id", "chn_features"});
    //  .capturePlanNodeId(charNameNodeId)
    tabel2Columns["chn"] = {"chn_id", "chn_features"};

    sources["chn"] = chn_a;

    auto ci_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["cast_info"]})
            .filter(
                "note = '(voice)' OR note = '(voice) (uncredited)' OR note = '(voice: English version)'")
            .project(
                {"movie_id",
                 "person_id",
                 "role_id",
                 "person_role_id",
                 "ci_features"});
    //  .capturePlanNodeId(castInfoNodeId)
    tabel2Columns["ci"] = {"movie_id",
                 "person_id",
                 "role_id",
                 "person_role_id",
                 "ci_features"};

    sources["ci"] = ci_a;

    auto cn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["company_name"]})
                    .filter("country_code ='[us]'")
                    .project({"id as cn_id", "cn_features"});
    //  .capturePlanNodeId(companyNameNodeId)
    tabel2Columns["cn"] = {"cn_id", "cn_features"};

    sources["cn"] = cn_a;

    auto it_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["info_type"]})
                    .filter("info = 'release dates'")
                    .project({"id as it_id", "it_features"});
    //  .capturePlanNodeId(infoTypeNodeId1)
    tabel2Columns["it"] = {"it_id", "it_features"};

    sources["it"] = it_a;

    auto it3_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["info_type"]})
                     .filter("info = 'trivia'")
                     .project({"id as it3_id", "it_features as it3_features"});
    //  .capturePlanNodeId(infoTypeNodeId2)
    tabel2Columns["it3"] = {"it3_id", "it3_features"};

    sources["it3"] = it3_a;

    auto k_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["keyword"]})
                   .filter("keyword = 'computer-animation'")
                   .project({"id as k_id", "k_features"});
    //  .capturePlanNodeId(keywordNodeId)
    tabel2Columns["k"] = {"k_id", "k_features"};

    sources["k"] = k_a;

    auto mc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_companies"]})
                    .project({"movie_id", "company_id", "mc_features"});
    //  .capturePlanNodeId(movieCompaniesNodeId)
    tabel2Columns["mc"] = {"movie_id", "company_id", "mc_features"};

    sources["mc"] = mc_a;

    auto mi_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["movie_info"]})
            .filter(
                "info IS NOT NULL AND (info LIKE 'Japan:%200%' OR info LIKE 'USA:%200%')")
            .project({"movie_id", "info_type_id", "mi_features"});
    //  .capturePlanNodeId(movieInfoNodeId)
    tabel2Columns["mi"] = {"movie_id", "info_type_id", "mi_features"};

    sources["mi"] = mi_a;

    auto mk_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_keyword"]})
                    .project({"movie_id", "keyword_id", "mk_features"});
    //  .capturePlanNodeId(movieKeywordNodeId)
    tabel2Columns["mk"] = {"movie_id", "keyword_id", "mk_features"};

    sources["mk"] = mk_a;

    auto n_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["name"]})
                   .filter("gender ='f' AND name LIKE '%An%'")
                   .project({"id as n_id", "n_features"});
    //  .capturePlanNodeId(nameNodeId)
    tabel2Columns["n"] = {"n_id", "n_features"};

    sources["n"] = n_a;

    auto pi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["person_info"]})
                    .project(
                        {"person_id as pi_person_id",
                         "info_type_id as pi_info_type_id",
                         "pi_features"});
    // .capturePlanNodeId(personInfoNodeId)
    tabel2Columns["pi"] = {"pi_person_id", "pi_info_type_id", "pi_features"};

    sources["pi"] = pi_a;

    auto rt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["role_type"]})
                    .filter("role ='actress'")
                    .project({"id as rt_id", "rt_features"});
    //  .capturePlanNodeId(roleTypeNodeId)
    tabel2Columns["rt"] = {"rt_id", "rt_features"};

    sources["rt"] = rt_a;

    auto t_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["title"]})
            .filter(
                "title = 'Shrek 2' AND  production_year > 2000 AND production_year < 2010")
            .project({"id", "t_features"});
    //    .capturePlanNodeId(titleNodeId)
    tabel2Columns["t"] = {"id", "t_features"};

    sources["t"] = t_a;

    return sources;
  }

  bool constructSQLQuery29(
      PlanBuilder& planBuilder,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      std::string subQueryId,
      std::string joinOrderStr) {
    /*
     * FROM aka_name AS an,
     complete_cast AS cc,
     comp_cast_type AS cct1,
     comp_cast_type AS cct2,
     char_name AS chn,
     cast_info AS ci,
     company_name AS cn,
     info_type AS it,
     info_type AS it3,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     name AS n,
     person_info AS pi,
     role_type AS rt,
     title AS t
     */

    core::PlanNodeId akaNameScanNodeId;
    core::PlanNodeId completeCastNodeId1;
    core::PlanNodeId completeCastNodeId2;
    core::PlanNodeId charNameNodeId;
    core::PlanNodeId castInfoNodeId;
    core::PlanNodeId companyNameNodeId;
    core::PlanNodeId infoTypeNodeId1;
    core::PlanNodeId infoTypeNodeId2;
    core::PlanNodeId keywordNodeId;
    core::PlanNodeId movieCompaniesNodeId;
    core::PlanNodeId movieInfoNodeId;
    core::PlanNodeId movieKeywordNodeId;
    core::PlanNodeId nameNodeId;
    core::PlanNodeId personInfoNodeId;
    core::PlanNodeId roleTypeNodeId;
    core::PlanNodeId titleNodeId;

    // filters
    /*
     * cct1.kind ='cast'
     AND cct2.kind ='complete+verified'
     AND chn.name = 'Queen'
     AND ci.note IN ('(voice)',
                  '(voice) (uncredited)',
                  '(voice: English version)')
     AND cn.country_code ='[us]'
     AND it.info = 'release dates'
     AND it3.info = 'trivia'
     AND k.keyword = 'computer-animation'
     AND mi.info IS NOT NULL
     AND (mi.info LIKE 'Japan:%200%'
       OR mi.info LIKE 'USA:%200%')
     AND n.gender ='f'
     AND n.name LIKE '%An%'
     AND rt.role ='actress'
     AND t.title = 'Shrek 2'
     AND t.production_year BETWEEN 2000 AND 2010
     */

    // projections
    /*
     * SELECT MIN(chn.name) AS voiced_char,
      MIN(n.name) AS voicing_actress,
      MIN(t.title) AS voiced_animation
     */

    std::unordered_map<std::string, PlanBuilder>
        sources; // with filters and projections pushed down;

    auto an_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["aka_name"]})
                    .project({"person_id as an_person_id", "an_features"});
    //  .capturePlanNodeId(akaNameNodeId1)

    sources["an_a"] = an_a;

    auto cc_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["complete_cast"]})
            .project({"movie_id", "subject_id", "status_id", "cc_features"});
    //  .capturePlanNodeId(completeCastNodeId1)

    sources["cc_a"] = cc_a;

    auto cct1_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .filter("kind = 'cast'")
            .project({"id as cct1_id", "cct_features as cct1_features"});
    //  .capturePlanNodeId(completeCastNodeId1)

    sources["cct1_a"] = cct1_a;

    auto cct2_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["comp_cast_type"]})
            .filter("kind ='complete+verified'")
            .project({"id as cct2_id", "cct_features as cct2_features"});
    //  .capturePlanNodeId(compCastTypeNodeId2)

    sources["cct2_a"] = cct2_a;

    auto chn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["char_name"]})
                     .filter("name = 'Queen'")
                     .project({"id as chn_id", "chn_features"});
    //  .capturePlanNodeId(charNameNodeId)

    sources["chn_a"] = chn_a;

    auto ci_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["cast_info"]})
            .filter("note = '(voice)' OR note = '(voice) (uncredited)' OR note = '(voice: English version)'")
            .project(
                {"movie_id",
                 "person_id",
                 "role_id",
                 "person_role_id",
                 "ci_features"});
    //  .capturePlanNodeId(castInfoNodeId)

    sources["ci_a"] = ci_a;

    auto cn_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["company_name"]})
                    .filter("country_code ='[us]'")
                    .project({"id as cn_id", "cn_features"});
    //  .capturePlanNodeId(companyNameNodeId)

    sources["cn_a"] = cn_a;

    auto it_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["info_type"]})
                    .filter("info = 'release dates'")
                    .project({"id as it_id", "it_features"});
    //  .capturePlanNodeId(infoTypeNodeId1)

    sources["it_a"] = it_a;

    auto it3_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .values({tableName2RowVector["info_type"]})
                     .filter("info = 'trivia'")
                     .project({"id as it3_id", "it_features as it3_features"});
    //  .capturePlanNodeId(infoTypeNodeId2)

    sources["it3_a"] = it3_a;

    auto k_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["keyword"]})
                   .filter("keyword = 'computer-animation'")
                   .project({"id as k_id", "k_features"});
    //  .capturePlanNodeId(keywordNodeId)

    sources["k_a"] = k_a;

    auto mc_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_companies"]})
                    .project({"movie_id", "company_id", "mc_features"});
    //  .capturePlanNodeId(movieCompaniesNodeId)

    sources["mc_a"] = mc_a;

    auto mi_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["movie_info"]})
            .filter(
                "info IS NOT NULL AND (info LIKE 'Japan:%200%' OR info LIKE 'USA:%200%')")
            .project({"movie_id", "info_type_id", "mi_features"});
    //  .capturePlanNodeId(movieInfoNodeId)

    sources["mi_a"] = mi_a;

    auto mk_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["movie_keyword"]})
                    .project({"movie_id", "keyword_id", "mk_features"});
    //  .capturePlanNodeId(movieKeywordNodeId)

    sources["mk_a"] = mk_a;

    auto n_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .values({tableName2RowVector["name"]})
                   .filter("gender ='f' AND name LIKE '%An%'")
                   .project({"id as n_id", "n_features"});
    //  .capturePlanNodeId(nameNodeId)

    sources["n_a"] = n_a;

    auto pi_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["person_info"]})
                    .project(
                        {"person_id as pi_person_id",
                         "info_type_id as pi_info_type_id",
                         "pi_features"});
    // .capturePlanNodeId(personInfoNodeId)

    sources["pi_a"] = pi_a;

    auto rt_a = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .values({tableName2RowVector["role_type"]})
                    .filter("role ='actress'")
                    .project({"id as rt_id", "rt_features"});
    //  .capturePlanNodeId(roleTypeNodeId)

    sources["rt_a"] = rt_a;

    auto t_a =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .values({tableName2RowVector["title"]})
            .filter("title = 'Shrek 2' AND  production_year > 2000 AND production_year < 2010")
            .project({"id", "t_features"});
    //    .capturePlanNodeId(titleNodeId)

    sources["t_a"] = t_a;

    // joins
    /*
     * AND t.id = mi.movie_id
     AND t.id = mc.movie_id
     AND t.id = ci.movie_id
     AND t.id = mk.movie_id
     AND t.id = cc.movie_id
     AND mc.movie_id = ci.movie_id
     AND mc.movie_id = mi.movie_id
     AND mc.movie_id = mk.movie_id
     AND mc.movie_id = cc.movie_id
     AND mi.movie_id = ci.movie_id
     AND mi.movie_id = mk.movie_id
     AND mi.movie_id = cc.movie_id
     AND ci.movie_id = mk.movie_id
     AND ci.movie_id = cc.movie_id
     AND mk.movie_id = cc.movie_id
     AND cn.id = mc.company_id
     AND it.id = mi.info_type_id
     AND n.id = ci.person_id
     AND rt.id = ci.role_id
     AND n.id = an.person_id
     AND ci.person_id = an.person_id
     AND chn.id = ci.person_role_id
     AND n.id = pi.person_id
     AND ci.person_id = pi.person_id
     AND it3.id = pi.info_type_id
     AND k.id = mk.keyword_id
     AND cct1.id = cc.subject_id
     AND cct2.id = cc.status_id;
     */

    // Create a JSON reader and root object
    Json::CharReaderBuilder readerBuilder;
    Json::Value root; // Root will hold the parsed JSON array
    std::string errors;

    // Parse the JSON string
    std::istringstream stream(joinOrderStr);
    if (!Json::parseFromStream(readerBuilder, stream, &root, &errors)) {
      std::cerr << "Error parsing JSON: " << errors << "\n";
      return 1;
    }

    std::string outName;
    std::string leftName;
    std::string rightName;
    std::vector<std::string> projections;
    // Iterate through the array
    for (const auto& item : root) {
      std::cout << "ID: " << item["ID"].asString() << "\n";
      std::cout << "Left: " << item["Left"].asString() << "\n";
      std::cout << "Right: " << item["Right"].asString() << "\n";
      std::cout << "Pred: " << item["Pred"].asString() << "\n";
      std::cout << "ProbeKeys: " << item["ProbeKeys"].asString() << "\n";
      std::cout << "BuildKeys: " << item["BuildKeys"].asString() << "\n";

      projections.clear();

      // Access Projection if it exists
      if (item.isMember("Projection")) {
        std::cout << "Projection: ";
        for (const auto& proj : item["Projection"]) {
          std::cout << proj << std::endl;
          projections.push_back(proj.asString());
        }
      }

      PlanBuilder left, right, out;

      // compose join
      leftName = item["Left"].asString() + "_" + subQueryId;

      // retrieve the corresponding PlanBuilder

      if (sources.count(leftName) > 0) {
        left = sources[leftName];
      }

      rightName = item["Right"].asString() + "_" + subQueryId;

      if (sources.count(rightName) > 0) {
        right = sources[rightName];
      }

      out = left.hashJoin(
          {item["ProbeKeys"].asString()},
          {item["BuildKeys"].asString()},
          right.planNode(),
          "",
          {projections});

      outName = item["ID"].asString() + "_" + subQueryId;
      sources[outName] = out;
    }
    std::string projString = "";
    bool isFirst = true;
    for (std::string proj : projections) {
      if (isFirst) {
        projString += proj;
        isFirst = false;
      } else {
        projString += "," + proj;
      }
    }
    projString = "concat(" + projString + ") as features";

    std::cout << "projString:" << projString << std::endl;

    planBuilder = sources[outName].project({projString});

    std::cout << "Plan: " << planBuilder.planNode()->toString(true, true)
              << std::endl;

    return true;
  }

  bool constructSQLQuery(
      int queryId,
      PlanBuilder& planBuilder,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator,
      std::string subqueryId) {
    switch (queryId) {
      case 29:

        std::string jsonString = R"([
            {"ID":"0", "Left":"t", "Right":"mi", "Pred":"t.id = mi.movie_id", "ProbeKeys":"id", "BuildKeys":"movie_id", "Projection":["id", "info_type_id", "t_features", "mi_features"]},
            {"ID":"1", "Left":"0", "Right":"mc", "Pred":"t.id = mc.movie_id", "ProbeKeys":"id", "BuildKeys":"movie_id", "Projection":["id", "info_type_id", "company_id", "t_features", "mi_features", "mc_features"]},
            {"ID":"2", "Left":"1", "Right":"ci", "Pred":"t.id = ci.movie_id", "ProbeKeys":"id", "BuildKeys":"movie_id", "Projection":["id", "info_type_id", "company_id", "person_id", "role_id", "person_role_id", "t_features", "mi_features", "mc_features", "ci_features"]},
            {"ID":"3", "Left":"2", "Right":"mk", "Pred":"t.id = mk.movie_id", "ProbeKeys":"id", "BuildKeys":"movie_id", "Projection":["id", "info_type_id", "company_id", "person_id", "role_id", "person_role_id", "keyword_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features"]},
            {"ID":"4", "Left":"3", "Right":"cc", "Pred":"t.id = cc.movie_id", "ProbeKeys":"id", "BuildKeys":"movie_id", "Projection":["id", "info_type_id", "company_id", "person_id", "role_id", "person_role_id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features"]},
            {"ID":"5", "Left":"4", "Right":"cn", "Pred":"cn.id (renamed to cn_id) = mc.company_id", "ProbeKeys":"company_id", "BuildKeys":"cn_id", "Projection":["id", "info_type_id", "person_id", "role_id", "person_role_id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features"]},
            {"ID":"6", "Left":"5", "Right":"it", "Pred":"it.id (renamed to it_id) = mi.info_type_id", "ProbeKeys":"info_type_id", "BuildKeys":"it_id", "Projection":["id", "person_id", "role_id", "person_role_id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features"]},
            {"ID":"7", "Left":"6", "Right":"n", "Pred":"n.id (renamed to n_id) = ci.person_id", "ProbeKeys":"person_id", "BuildKeys":"n_id", "Projection":["id", "person_id", "role_id", "person_role_id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features"]},
            {"ID":"8", "Left":"7", "Right":"rt", "Pred":"rt.id (renamed to rt_id) = ci.role_id", "ProbeKeys":"role_id", "BuildKeys":"rt_id", "Projection":["id", "person_id", "person_role_id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features"]},
            {"ID":"9", "Left":"8", "Right":"an", "Pred":"n.id (mapped to person_id) = an.person_id", "ProbeKeys":"person_id", "BuildKeys":"an_person_id", "Projection":["id", "person_id", "person_role_id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features", "an_features"]},
            {"ID":"10", "Left":"9", "Right":"chn", "Pred":"chn.id (renamed to chn_id) = ci.person_role_id", "ProbeKeys":"person_role_id", "BuildKeys":"chn_id", "Projection":["id", "person_id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features", "an_features", "chn_features"]},
            {"ID":"11", "Left":"10", "Right":"pi", "Pred":"n.id (mapped to person_id) = pi.person_id", "ProbeKeys":"person_id", "BuildKeys":"pi_person_id", "Projection":["id", "keyword_id", "subject_id", "status_id", "pi_info_type_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features", "an_features", "chn_features", "pi_features"]},
            {"ID":"12", "Left":"11", "Right":"it3", "Pred":"it3.id (renamed to it3_id) = pi.info_type_id", "ProbeKeys":"pi_info_type_id", "BuildKeys":"it3_id", "Projection":["id", "keyword_id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features", "an_features", "chn_features", "pi_features", "it3_features"]},
            {"ID":"13", "Left":"12", "Right":"k", "Pred":"k.id (renamed to k_id) = mk.keyword_id", "ProbeKeys":"keyword_id", "BuildKeys":"k_id", "Projection":["id", "subject_id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features", "an_features", "chn_features", "pi_features", "it3_features", "k_features"]},
            {"ID":"14", "Left":"13", "Right":"cct1", "Pred":"cct1.id (renamed to cct1_id) = cc.subject_id", "ProbeKeys":"subject_id", "BuildKeys":"cct1_id", "Projection":["id", "status_id", "t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features", "an_features", "chn_features", "pi_features", "it3_features", "k_features", "cct1_features"]},
            {"ID":"15", "Left":"14", "Right":"cct2", "Pred":"cct2.id (renamed to cct2_id) = cc.status_id", "ProbeKeys":"status_id", "BuildKeys":"cct2_id", "Projection":["t_features", "mi_features", "mc_features", "ci_features", "mk_features", "cc_features", "cn_features", "it_features", "n_features", "rt_features", "an_features", "chn_features", "pi_features", "it3_features", "k_features", "cct1_features", "cct2_features"]}
])";

        return constructSQLQuery29(
            planBuilder, planNodeIdGenerator, "a", jsonString);
    }
    return true;
  }

  float* genWeight(int dim1, int dim2) {
    int total_size = dim1 * dim2;

    // generate weight matrix
    float* weight = new float[total_size];
    for (int i = 0; i < total_size; i++) {
      if (i % 2 == 0) {
        weight[i] = 1.0;
      } else {
        weight[i] = 0.0;
      }
    }
    return weight;
  }

  bool addModelInferenceToQueryPlan(
      int modelId,
      int numInputFeatures,
      PlanBuilder& planBuilder,
      std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator) {
    switch (modelId) {
      case 0:

        NNModelContext context;
        sampleNNModelArch(numInputFeatures, context);
        int numLayers = context.numLayers;
        int hiddenLayerNeurons = context.hiddenLayerNeurons;
        int outputLayerNeurons = context.outputLayerNeurons;

        std::cout << "numLayers: " << numLayers << std::endl;
        std::cout << "hiddenLayerNeurons: " << hiddenLayerNeurons << std::endl;
        std::cout << "outputLayerNeurons: " << outputLayerNeurons << std::endl;

        std::vector<float*> biasMatrice;
        biasMatrice.push_back(genWeight(1, hiddenLayerNeurons));
        for (int i = 0; i < numLayers - 2; i++) {
          biasMatrice.push_back(genWeight(1, hiddenLayerNeurons));
        }
        biasMatrice.push_back(genWeight(1, outputLayerNeurons));

        std::vector<float*> weightMatrice;
        weightMatrice.push_back(
            genWeight(numInputFeatures, hiddenLayerNeurons));
        for (int i = 0; i < numLayers - 2; i++) {
          weightMatrice.push_back(
              genWeight(hiddenLayerNeurons, hiddenLayerNeurons));
        }
        weightMatrice.push_back(
            genWeight(numInputFeatures, outputLayerNeurons));

        NNBuilder builder = NNBuilder().denseLayer(
            hiddenLayerNeurons,
            numInputFeatures,
            weightMatrice[0],
            biasMatrice[0],
            NNBuilder::RELU);
        for (int i = 0; i < numLayers - 2; i++) {
          builder = builder.denseLayer(
              hiddenLayerNeurons,
              hiddenLayerNeurons,
              weightMatrice[i + 1],
              biasMatrice[i + 1],
              NNBuilder::RELU);
        }
        builder = builder.denseLayer(
            outputLayerNeurons,
            hiddenLayerNeurons,
            weightMatrice[numLayers - 1],
            biasMatrice[numLayers - 1],
            NNBuilder::SOFTMAX);
        std::string compute = builder.build();

        std::cout << compute << std::endl;

        // Fully connected neural network
        planBuilder.project({fmt::format(compute, "features")});

        break;
    }

    return true;
  }

  bool createAndExecuteQuery(int queryId, int modelId) {
    PlanBuilder planBuilder{pool_.get()};
    std::shared_ptr<core::PlanNodeIdGenerator> planNodeIdGenerator =
        std::make_shared<core::PlanNodeIdGenerator>();
    std::cout << "Step 1. Constructing query..." << std::endl;
    constructSQLQuery(29, planBuilder, planNodeIdGenerator, "a");
    std::cout << "Step 2. Add model inference to query plan" << std::endl;
    addModelInferenceToQueryPlan(
        modelId, 136, planBuilder, planNodeIdGenerator);
    auto myPlan = planBuilder.planNode();
    std::cout << myPlan->toString(true, true) << std::endl;
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    auto results =
        exec::test::AssertQueryBuilder(myPlan).copyResults(pool_.get());
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    std::cout << "Time (sec) = "
              << (std::chrono::duration_cast<std::chrono::microseconds>(
                      end - begin)
                      .count()) /
            1000000.0
              << std::endl;
    std::cout << "Results:" << results->toString() << std::endl;
    std::cout << results->toString(0, results->size()) << std::endl;
    return true;
  }
};

int main(int argc, char** argv) {
  setlocale(LC_TIME, "C");

  folly::init(&argc, &argv, false);
  memory::MemoryManager::initialize({});

  JobBenchmark bench;
  std::cout
      << "[WARNING] the data path is hardcoded and needs to be modified accordingly."
      << std::endl;

  // Load Aka Name table
  RowVectorPtr akaNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/aka_name.csv", "aka_name", 8);

  // Load Aka Title table
  RowVectorPtr akaTitleVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/aka_title.csv", "aka_title", 8);

  // Load Cast Info table
  RowVectorPtr castInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/cast_info.csv", "cast_info", 8);

  // Load Char Name table
  RowVectorPtr charNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/char_name.csv", "char_name", 8);

  // Load Comp Cast Type table
  RowVectorPtr compCastTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/comp_cast_type.csv", "comp_cast_type", 8);

  // Load Company Name table
  RowVectorPtr companyNameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/company_name.csv", "company_name", 8);

  // Load Company Type table
  RowVectorPtr companyTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/company_type.csv", "company_type", 8);

  // Load Complete Cast table
  RowVectorPtr completeCastVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/complete_cast.csv", "complete_cast", 8);

  // Load Info Type table
  RowVectorPtr infoTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/info_type.csv", "info_type", 8);

  // Load Keyword table
  RowVectorPtr keywordVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/keyword.csv", "keyword", 8);

  // Load Kind Type table
  RowVectorPtr kindTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/kind_type.csv", "kind_type", 8);

  // Load Link Type table
  RowVectorPtr linkTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/link_type.csv", "link_type", 8);

  // Load Movie Companies table
  RowVectorPtr movieCompaniesVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_companies.csv", "movie_companies", 8);

  // Load Movie Info Index table
  RowVectorPtr movieInfoIdxVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_info_idx.csv", "movie_info_idx", 8);

  // Load Movie Keyword table
  RowVectorPtr movieKeywordVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_keyword.csv", "movie_keyword", 8);

  // Load Movie Link table
  RowVectorPtr movieLinkVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_link.csv", "movie_link", 8);

  // Load Name table
  RowVectorPtr nameVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/name.csv", "name", 8);

  // Load Role Type table
  RowVectorPtr roleTypeVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/role_type.csv", "role_type", 8);

  // Load Title table
  RowVectorPtr titleVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/title.csv", "title", 8);

  // Load Movie Info table
  RowVectorPtr movieInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb/movie_info.csv", "movie_info", 8);

  // Load Person Info table
  RowVectorPtr personInfoVec = bench.getTableFromCSVFile(
      bench.maker, "resources/data/imdb//person_info.csv", "person_info", 8);

  bench.tableName2RowVector["aka_name"] = akaNameVec;

  bench.tableName2RowVector["aka_title"] = akaTitleVec;

  bench.tableName2RowVector["cast_info"] = castInfoVec;

  bench.tableName2RowVector["char_name"] = charNameVec;

  bench.tableName2RowVector["comp_cast_type"] = compCastTypeVec;

  bench.tableName2RowVector["company_name"] = companyNameVec;

  bench.tableName2RowVector["company_type"] = companyTypeVec;

  bench.tableName2RowVector["complete_cast"] = completeCastVec;

  bench.tableName2RowVector["info_type"] = infoTypeVec;

  bench.tableName2RowVector["keyword"] = keywordVec;

  bench.tableName2RowVector["kind_type"] = kindTypeVec;

  bench.tableName2RowVector["link_type"] = linkTypeVec;

  bench.tableName2RowVector["movie_companies"] = movieCompaniesVec;

  bench.tableName2RowVector["movie_info_idx"] = movieInfoIdxVec;

  bench.tableName2RowVector["movie_keyword"] = movieKeywordVec;

  bench.tableName2RowVector["movie_link"] = movieLinkVec;

  bench.tableName2RowVector["name"] = nameVec;

  bench.tableName2RowVector["role_type"] = roleTypeVec;

  bench.tableName2RowVector["title"] = titleVec;

  bench.tableName2RowVector["movie_info"] = movieInfoVec;

  bench.tableName2RowVector["person_info"] = personInfoVec;

  int queryId = bench.sampleQuery();

  int modelId = bench.sampleModel();

  bool ret = bench.createAndExecuteQuery(queryId, modelId);

  return ret;
}

