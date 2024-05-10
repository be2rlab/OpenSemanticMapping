// Source file for the surfel info program



////////////////////////////////////////////////////////////////////////
// Include files
////////////////////////////////////////////////////////////////////////

namespace gaps {}
using namespace gaps;
#include "R3Surfels/R3Surfels.h"



////////////////////////////////////////////////////////////////////////
// Program arguments
////////////////////////////////////////////////////////////////////////

static const char *input_scene_name = NULL;
static const char *input_database_name = NULL;
static int print_comments = 0;
static int print_features = 0;
static int print_scans = 0;
static int print_images = 0;
static int print_objects = 0;
static int print_labels = 0;
static int print_label_list = 0;
static int print_object_properties = 0;
static int print_label_properties = 0;
static int print_object_relationships = 0;
static int print_label_relationships = 0;
static int print_label_assignments = 0;
static int print_label_assignment_counts = 0;
static int print_tree = 0;
static int print_nodes = 0;
static int print_database = 0;
static int print_blocks = 0;
static int print_surfels = 0;
static const char *query_name = NULL;
static const char *accuracy_arff_name = NULL;



////////////////////////////////////////////////////////////////////////
// Surfel database I/O Functions
////////////////////////////////////////////////////////////////////////

static R3SurfelScene *
OpenScene(const char *input_scene_name, const char *input_database_name)
{
  // Allocate surfel scene
  R3SurfelScene *scene = new R3SurfelScene();
  if (!scene) {
    RNFail("Unable to allocate scene\n");
    return NULL;
  }

  // Open surfel scene files
  if (!scene->OpenFile(input_scene_name, input_database_name, "r", "r")) {
    delete scene;
    return NULL;
  }

  // Return scene
  return scene;
}



static int
CloseScene(R3SurfelScene *scene)
{
  // Close surfel scene
  if (!scene->CloseFile()) return 0;

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Print accuracy vs. arff file
////////////////////////////////////////////////////////////////////////

static int 
PrintAccuracy(R3SurfelScene *scene, const char *arff_filename)
{
#if 0
  // Open file
  FILE *fp = fopen(arff_filename, "r");
  if (!fp) {
    RNFail("Unable to open ARFF file %s\n", arff_filename);
    return 0;
  }

  // Initialize statistics
  int ncorrect = 0;
  int nincorrect = 0;
  int nunlabeled = 0;
  int nobjects = 0;

  // Parse file
  static char buffer [ 1024 * 1024 ];
  double position[3], axis[3][3], plane[4], color[3];
  while (fgets(buffer, 1024 * 1024, fp)) {
    if (!strncmp(buffer, "% Object", 8)) {
      // Parse basic stuff in comment
      /* char *percent_keyword = */ strtok(buffer, " \t\n");
      /* char *object_keyword = */ strtok(NULL, " \t\n");
      /* int id = */ atoi(strtok(NULL, " \t\n"));
      position[0] = atof(strtok(NULL, " \t\n"));
      position[1] = atof(strtok(NULL, " \t\n"));
      position[2] = atof(strtok(NULL, " \t\n"));
      axis[0][0] = atof(strtok(NULL, " \t\n"));
      axis[0][1] = atof(strtok(NULL, " \t\n"));
      axis[0][2] = atof(strtok(NULL, " \t\n"));
      axis[1][0] = atof(strtok(NULL, " \t\n"));
      axis[1][1] = atof(strtok(NULL, " \t\n"));
      axis[1][2] = atof(strtok(NULL, " \t\n"));
      axis[2][0] = atof(strtok(NULL, " \t\n"));
      axis[2][1] = atof(strtok(NULL, " \t\n"));
      axis[2][2] = atof(strtok(NULL, " \t\n"));
      plane[0] = atof(strtok(NULL, " \t\n"));
      plane[1] = atof(strtok(NULL, " \t\n"));
      plane[2] = atof(strtok(NULL, " \t\n"));
      plane[3] = atof(strtok(NULL, " \t\n"));
      color[0] = atof(strtok(NULL, " \t\n"));
      color[1] = atof(strtok(NULL, " \t\n"));
      color[2] = atof(strtok(NULL, " \t\n"));

      // Parse feature info in comment
      int nfeatures = atoi(strtok(NULL, " \t\n"));
      for (int i = 0; i < nfeatures; i++) {
        int feature_type = atoi(strtok(NULL, " \t\n"));
        if (feature_type == 1) {
          // Spin image
          /* double radius = */ atof(strtok(NULL, " \t\n"));
          /* double height = */ atof(strtok(NULL, " \t\n"));
          /* int nshells = */ atoi(strtok(NULL, " \t\n"));
          /* int nslices = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 2) {
          // Template
          /* double radius = */ atof(strtok(NULL, " \t\n"));
          /* double height = */ atof(strtok(NULL, " \t\n"));
          /* double spacing = */ atof(strtok(NULL, " \t\n"));
          /* int ntemplates = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 3) {
          // Oracle
          /* int parent1 = */ atoi(strtok(NULL, " \t\n"));
          /* int parent2 = */ atoi(strtok(NULL, " \t\n"));
          /* int parent3 = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 4) {
          // Grid
          /* int nvalues = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 5) {
          // Point set
          // Nothing
        }
        else if (feature_type == 6) {
          // Shape context
          /* double radius = */ atof(strtok(NULL, " \t\n"));
          /* double height = */ atof(strtok(NULL, " \t\n"));
          /* int nshells = */ atoi(strtok(NULL, " \t\n"));
          /* int nslices = */ atoi(strtok(NULL, " \t\n"));
          /* int nsectors = */ atoi(strtok(NULL, " \t\n"));
        }
        else if (feature_type == 7) {
          // Segmentation info
          // Nothing
        }
       else if (feature_type == 8) {
          // Alignments
          // Nothing
        }
      }

      // Parse assignments
      int label_id = -1;
      double confidence = 0;
      int nassignments = atoi(strtok(NULL, " \t\n"));
      for (int i = 0; i < nassignments; i++) {
        label_id = atoi(strtok(NULL, " \t\n"));
        /* char *label_name = */ strtok(NULL, " \t\n");
        confidence = atof(strtok(NULL, " \t\n"));
      }

      // Get object
      char object_name[1024];
      sprintf(object_name, "%d_%.3f_%.3f_%.3f", label_id, position[0], position[1], position[2]);
      R3SurfelObject *object = scene->FindObjectByName(object_name);
      if (!object) {
        RNFail("Unable to find object %s\n", object_name);
        return 0;
      }

      // Update statistics
      if (label_id >= 0) {
        if (object->NLabelAssignments() > 0) {
          // Check if label matches
          R3SurfelLabelAssignment *assignment = object->LabelAssignment(0);
          R3SurfelLabel *scene_label = assignment->Label();

          // Search for match to arff label
          int correct = 0;
          LPLabelSet *label_set = LPCurrentLabelSDatabase()->NSurfels()et();
          while (1) {
            if (label_id == scene_label->Identifier()) { correct = 1; break; }
            if (!label_set->Label(label_id)->Parent()) break;
            if (label_set->Label(label_id)->Parent() == label_set->Label(label_id)) break;
            label_id = label_set->Label(label_id)->Parent()->ID();
          } 
          printf("%40s %d\n", scene_label->Name(), correct);
          if (correct) ncorrect++;
          else nincorrect++;
        }
        else {
          nunlabeled++;
        }
        nobjects++;
      }
    }
  }

  // Close file
  fclose(fp);

  // Print stats
  printf("Label Accuracy:\n");
  printf("  # Correct = %d ( %.1f %%)\n", ncorrect, 100.0 * ncorrect / nobjects);
  printf("  # Incorrect = %d ( %.1f %%)\n", nincorrect, 100.0 * nincorrect / nobjects);
  printf("  # Unlabeled = %d ( %.1f %%)\n", nunlabeled, 100.0 * nunlabeled / nobjects);
  printf("  # Total = %d ( %.1f %%)\n", nobjects, 100.0 * ncorrect / nobjects);
#endif

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Printing 
////////////////////////////////////////////////////////////////////////

static int
PrintInfo(R3SurfelScene *scene)
{
  // Get convenient variables
  R3SurfelTree *tree = scene->Tree();
  R3SurfelDatabase *database = tree->Database();

  // Print scene info
  const R3Box& bbox = scene->BBox();
  const R3Point& centroid = scene->Centroid();
  printf("Scene:\n");
  printf("  Name = %s\n", (scene->Name()) ? scene->Name() : "None");
  printf("  Version = %d.%d\n", database->MajorVersion(), database->MinorVersion());
  printf("  Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
  printf("  Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
  printf("  Axial lengths = ( %g %g %g )\n", bbox.XLength(), bbox.YLength(), bbox.ZLength());
  printf("  Max surfel identifier = %u\n", database->MaxIdentifier());
  printf("  # Comments = %d\n", scene->NComments());
  printf("  # Labels = %d\n", scene->NLabels());
  printf("  # Objects = %d\n", scene->NObjects());
  printf("  # Features = %d\n", scene->NFeatures());
  printf("  # Scans = %d\n", scene->NScans());
  printf("  # Images = %d\n", scene->NImages());
  printf("  # Object Properties = %d\n", scene->NObjectProperties());
  printf("  # Label Properties = %d\n", scene->NLabelProperties());
  printf("  # Object Relationships = %d\n", scene->NObjectRelationships());
  printf("  # Label Relationships = %d\n", scene->NLabelRelationships());
  printf("  # Label Assignments = %d\n", scene->NLabelAssignments());
  printf("  # Nodes = %d\n", tree->NNodes());
  printf("  # Blocks = %d\n", database->NBlocks());
  printf("  # Surfels in database = %lld\n", database->NSurfels());
  printf("  # Surfels in tree = %lld\n", tree->NSurfels(FALSE));
  printf("  # Surfels in leaves = %lld\n", tree->NSurfels(TRUE));
  printf("\n\n");
  
  // Print comment info
  if (print_comments) {
    printf("Comments:\n");
    for (int i = 0; i < scene->NComments(); i++) {
      const char *comment = scene->Comment(i);
      printf("  Comment %d = %s\n", i, comment);
      printf("\n");
    }
    printf("\n");
  }

  // Print label info
  if (print_labels) {
    printf("Labels:\n");
    RNArray<R3SurfelLabel *> stack;
    for (int i = 0; i < scene->NLabels(); i++) {
      R3SurfelLabel *label = scene->Label(i);
      if (!label->Parent()) stack.Insert(label);
    }
    while(!stack.IsEmpty()) {
      R3SurfelLabel *label = stack.Tail();
      stack.RemoveTail();
      for (int i = 0; i < label->NParts(); i++) stack.Insert(label->Part(i));
      if (query_name && (!label->Name() || strcmp(query_name, label->Name()))) continue;
      char prefix[16536];
      strncpy(prefix, " ", 16535);
      int level = label->PartHierarchyLevel();
      char assignment_keystroke = (label->AssignmentKeystroke() >= 0) ? label->AssignmentKeystroke() : ' ';
      for (int i = 0; i < level; i++) strncat(prefix, " ", 16535);
      printf("%s Label %d\n", prefix, label->SceneIndex());
      printf("%s Name = %s\n", prefix, (label->Name()) ? label->Name() : "None");
      printf("%s Identifier = %d\n", prefix, label->Identifier());
      printf("%s Assignment keystroke = %c\n", prefix, assignment_keystroke);
      printf("%s Part hierarchy level = %d\n", prefix, label->PartHierarchyLevel());
      printf("%s # Parts = %d\n", prefix, label->NParts());
      printf("%s # Label Properties = %d\n", prefix, label->NLabelProperties());
      printf("%s # Label Relationships = %d\n", prefix, label->NLabelRelationships());
      printf("%s # Assignments = %d\n", prefix, label->NLabelAssignments());
      printf("\n");
    }
    printf("\n");
  }

  // Print label list
  if (print_label_list) {
    printf("Label list:\n");
    for (int i = 0; i < scene->NLabels(); i++) {
      R3SurfelLabel *label = scene->Label(i);
      R3SurfelLabel *parent = label->Parent();
      int key = label->AssignmentKeystroke();
      if (query_name && (!label->Name() || strcmp(query_name, label->Name()))) continue;
      printf("  %-20s  %3d  %c  %-20s  1  %.3f %.3f %.3f\n",
        (label->Name()) ? label->Name() : "Null",
        (label->Identifier() >= 0) ? label->Identifier() : 0,
        ((key >= 32) && (key < 127)) ? key : '-',
        (parent && parent->Name()) ? parent->Name() : "Null",
        label->Color().R(), label->Color().G(), label->Color().B());
    }
  }

  // Print object info
  if (print_objects) {
    printf("Objects:\n");
    RNArray<R3SurfelObject *> stack;
    for (int i = 0; i < scene->NObjects(); i++) {
      R3SurfelObject *object = scene->Object(i);
      if (!object->Parent()) stack.Insert(object);
    }
    while(!stack.IsEmpty()) {
      R3SurfelObject *object = stack.Tail();
      stack.RemoveTail();
      for (int i = 0; i < object->NParts(); i++) stack.Insert(object->Part(i));
      if (query_name && (!object->Name() || strcmp(query_name, object->Name()))) continue;
      char prefix[16536];
      strncpy(prefix, " ", 16535);
      int level = object->PartHierarchyLevel();
      for (int i = 0; i < level; i++) strncat(prefix, " ", 16535);
      R3Box bbox = object->BBox();
      R3Point centroid = object->Centroid();
      R3OrientedBox obb = object->CurrentOrientedBBox();
      RNInterval timestamp_range = object->TimestampRange();
      R3SurfelLabel *predicted_label = object->PredictedLabel();
      R3SurfelLabel *ground_truth_label = object->GroundTruthLabel();
      // const R3SurfelFeatureVector& vector = object->FeatureVector();
      printf("%s Object %d\n", prefix, object->SceneIndex());
      printf("%s Name = %s\n", prefix, (object->Name()) ? object->Name() : "None");
      printf("%s Identifier = %d\n", prefix, object->Identifier());
      printf("%s Complexity = %g\n", prefix, object->Complexity());
      printf("%s Part hierarchy level = %d\n", prefix, object->PartHierarchyLevel());
      printf("%s Centroid = ( %g %g %g )\n", prefix, centroid[0], centroid[1], centroid[2]);
      if (!bbox.IsEmpty())
        printf("%s ABB = ( %g %g %g ) ( %g %g %g )\n", prefix, bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
      if (!obb.IsEmpty())
        printf("%s OBB = ( %g %g %g ) ( %g %g %g ) ( %g %g %g ) %g %g %g\n", prefix,
          obb.Center().X(), obb.Center().Y(), obb.Center().Z(),
          obb.Axis(0).X(), obb.Axis(0).Y(), obb.Axis(0).Z(),                
          obb.Axis(1).X(), obb.Axis(1).Y(), obb.Axis(1).Z(),                
          obb.Radius(0), obb.Radius(1), obb.Radius(2));               
      if (!timestamp_range.IsEmpty())
        printf("%s Timestamp Range = %.9f %.9f\n", prefix, timestamp_range.Min(), timestamp_range.Max());
      printf("%s # Nodes = %d\n", prefix, object->NNodes());
      printf("%s # Parts = %d\n", prefix, object->NParts());
      printf("%s # Object Properties = %d\n", prefix, object->NObjectProperties());
      printf("%s # Object Relationships = %d\n", prefix, object->NObjectRelationships());
      printf("%s # Assignments = %d\n", prefix, object->NLabelAssignments());
      printf("%s Predicted Label = %s\n", prefix, (predicted_label) ? predicted_label->Name() : "None");
      printf("%s Ground Truth Label = %s\n", prefix, (ground_truth_label) ? ground_truth_label->Name() : "None");
      printf("%s Flags = %u\n", prefix, (unsigned int) object->Flags());
      // printf("%s Feature Vector = ", prefix);
      // for (int i = 0; i < vector.NValues(); i++) printf("%12.6f ", vector.Value(i));
      // printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print label property info
  if (print_label_properties) {
    printf("Label Properties:\n");
    for (int i = 0; i < scene->NLabelProperties(); i++) {
      R3SurfelLabelProperty *property = scene->LabelProperty(i);
      R3SurfelLabel *label = property->Label();
      printf("  Label Property %d\n", i);
      printf("    Type = %d\n", property->Type());
      printf("    Label = %d\n", (label) ? label->SceneIndex() : -1);
      printf("    Operands = %d : ", property->NOperands());
      for (int j = 0; j < property->NOperands(); j++) printf("%12.6f ", property->Operand(j));
      printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print object property info
  if (print_object_properties) {
    printf("Object Properties:\n");
    for (int i = 0; i < scene->NObjectProperties(); i++) {
      R3SurfelObjectProperty *property = scene->ObjectProperty(i);
      R3SurfelObject *object = property->Object();
      printf("  Object Property %d\n", i);
      printf("    Type = %d\n", property->Type());
      printf("    Object = %d\n", (object) ? object->SceneIndex() : -1);
      printf("    Operands = %d : ", property->NOperands());
      for (int j = 0; j < property->NOperands(); j++) printf("%12.6f ", property->Operand(j));
      printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print label relationship info
  if (print_label_relationships) {
    printf("Label Relationships:\n");
    for (int i = 0; i < scene->NLabelRelationships(); i++) {
      R3SurfelLabelRelationship *relationship = scene->LabelRelationship(i);
      R3SurfelLabel *label0 = relationship->Label(0);
      R3SurfelLabel *label1 = relationship->Label(1);
      printf("  Label Relationship %d\n", i);
      printf("    Type = %d\n", relationship->Type());
      printf("    Label0 = %d\n", (label0) ? label0->SceneIndex() : -1);
      printf("    Label1 = %d\n", (label1) ? label1->SceneIndex() : -1);
      printf("    Operands = %d : ", relationship->NOperands());
      for (int j = 0; j < relationship->NOperands(); j++) printf("%12.6f ", relationship->Operand(j));
      printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print object relationship info
  if (print_object_relationships) {
    printf("Object Relationships:\n");
    for (int i = 0; i < scene->NObjectRelationships(); i++) {
      R3SurfelObjectRelationship *relationship = scene->ObjectRelationship(i);
      R3SurfelObject *object0 = relationship->Object(0);
      R3SurfelObject *object1 = relationship->Object(1);
      printf("  Object Relationship %d\n", i);
      printf("    Type = %d\n", relationship->Type());
      printf("    Object0 = %d\n", (object0) ? object0->SceneIndex() : -1);
      printf("    Object1 = %d\n", (object1) ? object1->SceneIndex() : -1);
      printf("    Operands = %d : ", relationship->NOperands());
      for (int j = 0; j < relationship->NOperands(); j++) printf("%12.6f ", relationship->Operand(j));
      printf("\n");
      printf("\n");
    }
    printf("\n");
  }

  // Print label assignment info
  if (print_label_assignments) {
    printf("Label Assignments:\n");
    for (int i = 0; i < scene->NLabelAssignments(); i++) {
      R3SurfelLabelAssignment *assignment = scene->LabelAssignment(i);
      R3SurfelObject *object = assignment->Object();
      R3SurfelLabel *label = assignment->Label();
      printf("  Label Assignment %d\n", i);
      printf("    Object = %d\n", (object) ? object->SceneIndex() : -1);
      printf("    Label = %d\n", (label) ? label->SceneIndex() : -1);
      printf("    Confidence = %g\n", assignment->Confidence());
      printf("    Originator = %d\n", assignment->Originator());
      printf("\n");
    }
    printf("\n");
  }

  // Print label assignment counts
  if (print_label_assignment_counts) {
    printf("Label Assignment Counts:\n");
    printf("  ASSIGNMENT_COUNTS ");
    static const int max_identifier = 255;
    for (int identifier = 0; identifier <= max_identifier; identifier++) {
      R3SurfelLabel *label = scene->FindLabelByIdentifier(identifier);
      if (!label) continue;
      if (label == scene->RootLabel()) continue;
      int top_level_assignments = 0;
      for (int j = 0; j < label->NLabelAssignments(); j++) {
        R3SurfelLabelAssignment *assignment = label->LabelAssignment(j);
        R3SurfelObject *object = assignment->Object();
        if (!object) continue;
        if (!object->Parent()) continue;
        if (object->Parent() != scene->RootObject()) continue;
        top_level_assignments++;
      }
      printf(" %d", top_level_assignments);
    }
    printf("\n");
  }

  // Print tree info
  if (print_tree) {
    const R3Box& bbox = tree->BBox();
    const R3Point& centroid = tree->Centroid();
    RNInterval timestamp_range = tree->TimestampRange();
    printf("Tree:\n");
    printf("  # Nodes = %d\n", tree->NNodes());
    printf("  Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
    printf("  Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
    printf("  Axial lengths = ( %g %g %g )\n", bbox.XLength(), bbox.YLength(), bbox.ZLength());
    printf("  Timestamp Range = %.9f %.9f\n", timestamp_range.Min(), timestamp_range.Max());
    printf("\n");
  }

  // Print node info
  if (print_nodes) {
    printf("Nodes:\n");
    RNArray<R3SurfelNode *> stack;
    R3SurfelNode *node = tree->RootNode();
    if (node) stack.Insert(node);
    while(!stack.IsEmpty()) {
      R3SurfelNode *node = stack.Tail();
      stack.RemoveTail();
      for (int i = 0; i < node->NParts(); i++) stack.Insert(node->Part(i));
      if (query_name && (!node->Name() || strcmp(query_name, node->Name()))) continue;
      char prefix[16536];
      strncpy(prefix, " ", 16535);
      int level = node->TreeLevel();
      for (int i = 0; i < level; i++) strncat(prefix, " ", 16535);
      R3Box bbox = node->BBox();
      R3Point centroid = node->Centroid();
      RNInterval timestamp_range = node->TimestampRange();
      printf("%s  Node %s\n", prefix, node->Name());
      printf("%s    # Parts = %d\n", prefix, node->NParts());
      printf("%s    # Blocks = %d\n", prefix, node->NBlocks());
      printf("%s    Object = %d\n", prefix, (node->Object()) ? node->Object()->SceneIndex() : -1);
      printf("%s    Scan = %d\n", prefix, (node->Scan()) ? node->Scan()->SceneIndex() : -1);
      printf("%s    Complexity = %g\n", prefix, node->Complexity());
      printf("%s    Resolution = %g\n", prefix, node->Resolution());
      printf("%s    Average Radius = %g\n", prefix, node->AverageRadius());
      printf("%s    Centroid = ( %g %g %g )\n", prefix, centroid[0], centroid[1], centroid[2]);
      printf("%s    Bounding box = ( %g %g %g ) ( %g %g %g )\n", prefix, bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
      printf("%s    Timestamp Range = %.9f %.9f\n", prefix, timestamp_range.Min(), timestamp_range.Max());
      printf("\n");
    }
    printf("\n");
  }

  // Print database info
  if (print_database) {
    const R3Box& bbox = database->BBox();
    const R3Point& centroid = database->Centroid();
    printf("Database:\n");
    printf("  # Blocks = %d\n", database->NBlocks());
    printf("  # Surfels = %lld\n", database->NSurfels());
    printf("  Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
    printf("  Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
    printf("  Axial lengths = ( %g %g %g )\n", bbox.XLength(), bbox.YLength(), bbox.ZLength());
    printf("  Surfel size = %lu\n", sizeof(R3Surfel));
    printf("\n");
  }

  // Print block info
  if (print_blocks) {
    printf("Blocks:\n");
    for (int i = 0; i < database->NBlocks(); i++) {
      R3SurfelBlock *block = database->Block(i);
      R3Point position_origin = block->PositionOrigin();
      RNInterval timestamp_range = block->TimestampRange();
      R3Box bbox = block->BBox();
      R3Point centroid = block->Centroid();
      printf("  Block %d\n", i);
      printf("    # Surfels = %d\n", block->NSurfels());
      printf("    Node = %d\n", (block->Node()) ? block->Node()->TreeIndex() : -1);
      printf("    Resolution = %g\n", block->Resolution());
      printf("    Average Radius = %g\n", block->AverageRadius());
      printf("    Position Origin = ( %g %g %g )\n", position_origin[0], position_origin[1], position_origin[2]);
      printf("    Centroid = ( %g %g %g )\n", centroid[0], centroid[1], centroid[2]);
      printf("    Bounding box = ( %g %g %g ) ( %g %g %g )\n", bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]);
      printf("    Timestamp Origin = %.9f\n", block->TimestampOrigin());
      printf("    Timestamp Range = %.9f %.9f\n", timestamp_range.Min(), timestamp_range.Max());
      printf("    Max surfel identifier = %u\n", block->MaxIdentifier());
      printf("\n");
    }
    printf("\n");
  }

  // Print surfel info
  if (print_surfels) {
    printf("Surfels:\n");
    for (int i = 0; i < tree->NNodes(); i++) {
      R3SurfelNode *node = tree->Node(i);
      for (int j = 0; j < node->NBlocks(); j++) {
        R3SurfelBlock *block = node->Block(j);
        database->ReadBlock(block);
        printf("  Block %d\n", i);
        printf("    # Surfels = %d\n", block->NSurfels());
        for (int j = 0; j < block->NSurfels(); j++) {
          const R3Surfel *surfel = block->Surfel(j);
          printf("    Surfel %d\n", j);
          printf("      Position = %f %f %f\n", surfel->PX(), surfel->PY(), surfel->PZ());
          printf("      Normal = %f %f %f\n", surfel->NX(), surfel->NY(), surfel->NZ());
          printf("      Tangent = %f %f %f\n", surfel->TX(), surfel->TY(), surfel->TZ());
          printf("      Color = %d %d %d\n", surfel->R(), surfel->G(), surfel->B());
          printf("      Radius = %f %f\n", surfel->Radius(0), surfel->Radius(1));
          printf("      Depth = %.6f\n", surfel->Depth());
          printf("      Elevation = %.6f\n", surfel->Elevation());
          printf("      Timestamp = %.9f\n", surfel->Timestamp());
          printf("      Identifier = %u\n", surfel->Identifier());
          printf("      Attribute = %u\n", surfel->Attribute());
          printf("      Flags = %d\n", surfel->Flags());
        }
        database->ReleaseBlock(block);
      }
      printf("\n");
    }
    printf("\n");
  }

  // Print feature info
  if (print_features) {
    printf("Features:\n");
    for (int i = 0; i < scene->NFeatures(); i++) {
      R3SurfelFeature *feature = scene->Feature(i);
      if (query_name && (!feature->Name() || strcmp(query_name, feature->Name()))) continue;
      printf("  Name = %s\n", (feature->Name()) ? feature->Name() : "None");
      printf("  Weight = %g\n", feature->Weight());
      printf("  Minimum = %g\n", feature->Minimum());
      printf("  Maximum = %g\n", feature->Maximum());
      printf("\n");
    }
    printf("\n");
  }

  // Print scan info
  if (print_scans) {
    printf("Scans:\n");
    for (int i = 0; i < scene->NScans(); i++) {
      R3SurfelScan *scan = scene->Scan(i);
      if (query_name && (!scan->Name() || strcmp(query_name, scan->Name()))) continue;
      printf("  Name = %s\n", (scan->Name()) ? scan->Name() : "None");
      printf("  Viewpoint = %g %g %g\n", scan->Viewpoint().X(), scan->Viewpoint().Y(), scan->Viewpoint().Z());
      printf("  Towards = %g %g %g\n", scan->Towards().X(), scan->Towards().Y(), scan->Towards().Z());
      printf("  Up = %g %g %g\n", scan->Up().X(), scan->Up().Y(), scan->Up().Z());
      printf("  Timestamp = %.9f\n", scan->Timestamp());
      printf("  Node = %d\n", (scan->Node()) ? scan->Node()->TreeIndex() : -1);
      printf("  Image = %d\n", (scan->Image()) ? scan->Image()->SceneIndex() : -1);
      printf("\n");
    }
    printf("\n");
  }

  // Print image info
  if (print_images) {
    printf("Images:\n");
    for (int i = 0; i < scene->NImages(); i++) {
      R3SurfelImage *image = scene->Image(i);
      if (query_name && (!image->Name() || strcmp(query_name, image->Name()))) continue;
      const RNScalar *radial_distortion = image->RadialDistortion();
      const RNScalar *tangential_distortion = image->TangentialDistortion();
      printf("  Name = %s\n", (image->Name()) ? image->Name() : "None");
      printf("  Scan = %d\n", (image->Scan()) ? image->Scan()->SceneIndex() : -1);
      printf("  Viewpoint = %g %g %g\n", image->Viewpoint().X(), image->Viewpoint().Y(), image->Viewpoint().Z());
      printf("  Towards = %g %g %g\n", image->Towards().X(), image->Towards().Y(), image->Towards().Z());
      printf("  Up = %g %g %g\n", image->Up().X(), image->Up().Y(), image->Up().Z());
      printf("  FOV = %g %g\n", image->XFOV(), image->YFOV());
      printf("  Image dimensions = %d %d\n", image->ImageWidth(), image->ImageHeight());
      printf("  Image center = %g %g\n", image->ImageCenter().X(), image->ImageCenter().Y());
      printf("  Focal lengths = %g %g\n", image->XFocal(), image->YFocal());
      printf("  Timestamp = %.9f\n", image->Timestamp());
      printf("  Distortion type = %d\n", image->DistortionType());
      if (image->DistortionType() != R3_SURFEL_NO_DISTORTION) {
        printf("  Radial distortion = %g %g %g\n", radial_distortion[0], radial_distortion[1], radial_distortion[2]);
        printf("  Tangential distortion = %g %g\n", tangential_distortion[0], tangential_distortion[1]);
      }
      if (image->HasRollingShutter()) {
        const RNScalar *rolling_shutter_timestamps = image->RollingShutterTimestamps();
        const R3CoordSystem *rolling_shutter_poses = image->RollingShutterPoses();
        R3Point viewpoint0 = rolling_shutter_poses[0].Matrix() * R3zero_point;
        R3Point viewpoint1 = rolling_shutter_poses[1].Matrix() * R3zero_point;
        R3Vector towards0 = rolling_shutter_poses[0].Matrix() * R3negz_vector;
        R3Vector towards1 = rolling_shutter_poses[1].Matrix() * R3negz_vector;
        R3Vector up0 = rolling_shutter_poses[0].Matrix() * R3posy_vector;
        R3Vector up1 = rolling_shutter_poses[1].Matrix() * R3posy_vector;
        printf("  Rolling shutter timestamps = %.9f %.9f\n",
           rolling_shutter_timestamps[0], rolling_shutter_timestamps[1]);
        printf("  Rolling shutter viewpoints= %g %g %g    %g %g %g\n",
           viewpoint0.X(), viewpoint0.Y(), viewpoint0.Z(), viewpoint1.X(), viewpoint1.Y(), viewpoint1.Z());
        printf("  Rolling shutter towards= %g %g %g    %g %g %g\n",
           towards0.X(), towards0.Y(), towards0.Z(), towards1.X(), towards1.Y(), towards1.Z());
        printf("  Rolling shutter ups= %g %g %g    %g %g %g\n",
           up0.X(), up0.Y(), up0.Z(), up1.X(), up1.Y(), up1.Z());
      }
      printf("\n");
    }
    printf("\n");
  }

  // Print accuracy info
  if (accuracy_arff_name) {
    PrintAccuracy(scene, accuracy_arff_name);
  }

  // Return success
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Argument Parsing Functions
////////////////////////////////////////////////////////////////////////

static int 
ParseArgs(int argc, char **argv)
{
  // Check number of program arguments
  if (argc < 3) {
    printf("Usage: sflinfo scenefile databasefile [options]\n");
    return 0;
  }

  // File names are first three arguments
  input_scene_name = argv[1];
  input_database_name = argv[2];

  // Parse arguments
  argc -= 3; argv += 3;
  while (argc > 0) {
    if (!strcmp(*argv, "-v")) { print_labels = 1; print_objects = 1; }
    else if (!strcmp(*argv, "-comments")) { print_comments = 1; }
    else if (!strcmp(*argv, "-features")) { print_features = 1; }
    else if (!strcmp(*argv, "-objects")) { print_objects = 1; }
    else if (!strcmp(*argv, "-labels")) { print_labels = 1; }
    else if (!strcmp(*argv, "-label_list")) { print_label_list = 1; }
    else if (!strcmp(*argv, "-properties")) { print_object_properties = print_label_properties = 1; }
    else if (!strcmp(*argv, "-object_properties")) { print_object_properties = 1; }
    else if (!strcmp(*argv, "-label_properties")) { print_label_properties = 1; }
    else if (!strcmp(*argv, "-object_relationships")) { print_object_relationships = 1; }
    else if (!strcmp(*argv, "-label_relationships")) { print_label_relationships = 1; }
    else if (!strcmp(*argv, "-assignments")) { print_label_assignments = 1; }
    else if (!strcmp(*argv, "-assignment_counts")) { print_label_assignment_counts = 1; }
    else if (!strcmp(*argv, "-tree")) { print_tree = 1; }
    else if (!strcmp(*argv, "-nodes")) { print_nodes = 1; }
    else if (!strcmp(*argv, "-database")) { print_database = 1; }
    else if (!strcmp(*argv, "-blocks")) { print_blocks = 1; }
    else if (!strcmp(*argv, "-surfels")) { print_surfels = 1; }
    else if (!strcmp(*argv, "-scans")) { print_scans = 1; }
    else if (!strcmp(*argv, "-images")) { print_images = 1; }
    else if (!strcmp(*argv, "-query")) { argc--; argv++; query_name = *argv; }
    else if (!strcmp(*argv, "-accuracy")) { argc--; argv++; accuracy_arff_name = *argv; }
    else { RNFail("Invalid program argument: %s", *argv); exit(1); }
    argv++; argc--;
  }

  // Return OK status 
  return 1;
}



////////////////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Parse program arguments
  if (!ParseArgs(argc, argv)) exit(-1);

  // Open scene file
  R3SurfelScene *scene = OpenScene(input_scene_name, input_database_name);
  if (!scene) exit(-1);

  // Print info
  if (!PrintInfo(scene)) exit(-1);

  // Close scene file
  if (!CloseScene(scene)) exit(-1);;

  // Delete scene
  delete scene;
  
  // Return success 
  return 0;
}


