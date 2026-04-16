import 'dart:async';
import 'dart:ui';

typedef DrawImageHandler = FutureOr<void> Function({
  required String fileName,
  required Offset origin,
  required double objectScale,
  String? boardObjectId,
  String? logicalName,
  String? processedId,
  Set<String>? aliases,
});

typedef WriteTextHandler = FutureOr<void> Function({
  required String prompt,
  required Offset origin,
  required double letterSize,
  double? strokeSlowdown,
  String? boardObjectId,
  String? logicalName,
  String? attachedToObjectId,
  Set<String>? aliases,
});

typedef DeleteObjectHandler = FutureOr<void> Function({
  String? id,
  String? name,
});

typedef MoveObjectHandler = FutureOr<void> Function({
  required String target,
  required Offset newOrigin,
});

typedef LinkToImageHandler = FutureOr<void> Function({
  required String target,
  required String imageName,
});

typedef DeleteSelfHandler = FutureOr<void> Function({
  required String target,
});

typedef DrawClusterHandler = FutureOr<void> Function({
  required String clusterRef,
});

typedef HighlightClusterHandler = FutureOr<void> Function({
  required String clusterRef,
});

typedef ZoomClusterHandler = FutureOr<void> Function({
  required String clusterRef,
});

typedef WriteLabelHandler = FutureOr<void> Function({
  required String clusterRef,
  required String text,
});

typedef ConnectClustersHandler = FutureOr<void> Function({
  required String fromClusterRef,
  required String toClusterRef,
});

class WhiteboardActions {
  const WhiteboardActions({
    required DrawImageHandler onDrawImage,
    required WriteTextHandler onWriteText,
    required DeleteObjectHandler onDeleteObject,
    required MoveObjectHandler onMoveObject,
    required LinkToImageHandler onLinkToImage,
    required DeleteSelfHandler onDeleteSelf,
    required DrawClusterHandler onDrawCluster,
    required HighlightClusterHandler onHighlightCluster,
    required ZoomClusterHandler onZoomCluster,
    required WriteLabelHandler onWriteLabel,
    required ConnectClustersHandler onConnectClusters,
  })  : _onDrawImage = onDrawImage,
        _onWriteText = onWriteText,
        _onDeleteObject = onDeleteObject,
        _onMoveObject = onMoveObject,
        _onLinkToImage = onLinkToImage,
        _onDeleteSelf = onDeleteSelf,
        _onDrawCluster = onDrawCluster,
        _onHighlightCluster = onHighlightCluster,
        _onZoomCluster = onZoomCluster,
        _onWriteLabel = onWriteLabel,
        _onConnectClusters = onConnectClusters;

  final DrawImageHandler _onDrawImage;
  final WriteTextHandler _onWriteText;
  final DeleteObjectHandler _onDeleteObject;
  final MoveObjectHandler _onMoveObject;
  final LinkToImageHandler _onLinkToImage;
  final DeleteSelfHandler _onDeleteSelf;
  final DrawClusterHandler _onDrawCluster;
  final HighlightClusterHandler _onHighlightCluster;
  final ZoomClusterHandler _onZoomCluster;
  final WriteLabelHandler _onWriteLabel;
  final ConnectClustersHandler _onConnectClusters;

  Future<void> drawImage({
    required String fileName,
    required Offset origin,
    required double objectScale,
    String? boardObjectId,
    String? logicalName,
    String? processedId,
    Set<String>? aliases,
  }) async {
    await _onDrawImage(
      fileName: fileName,
      origin: origin,
      objectScale: objectScale,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      processedId: processedId,
      aliases: aliases,
    );
  }

  Future<void> draw_image({
    required String fileName,
    required Offset origin,
    required double objectScale,
    String? boardObjectId,
    String? logicalName,
    String? processedId,
    Set<String>? aliases,
  }) {
    return drawImage(
      fileName: fileName,
      origin: origin,
      objectScale: objectScale,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      processedId: processedId,
      aliases: aliases,
    );
  }

  Future<void> writeText({
    required String prompt,
    required Offset origin,
    required double letterSize,
    double? strokeSlowdown,
    String? boardObjectId,
    String? logicalName,
    String? attachedToObjectId,
    Set<String>? aliases,
  }) async {
    await _onWriteText(
      prompt: prompt,
      origin: origin,
      letterSize: letterSize,
      strokeSlowdown: strokeSlowdown,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      attachedToObjectId: attachedToObjectId,
      aliases: aliases,
    );
  }

  Future<void> write_text({
    required String prompt,
    required Offset origin,
    required double letterSize,
    double? strokeSlowdown,
    String? boardObjectId,
    String? logicalName,
    String? attachedToObjectId,
    Set<String>? aliases,
  }) {
    return writeText(
      prompt: prompt,
      origin: origin,
      letterSize: letterSize,
      strokeSlowdown: strokeSlowdown,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      attachedToObjectId: attachedToObjectId,
      aliases: aliases,
    );
  }

  Future<void> deleteObject({
    String? id,
    String? name,
  }) async {
    await _onDeleteObject(id: id, name: name);
  }

  Future<void> delete_object({
    String? id,
    String? name,
  }) {
    return deleteObject(id: id, name: name);
  }

  Future<void> deleteImage({
    required String name,
  }) {
    return deleteObject(name: name);
  }

  Future<void> delete_image({
    required String name,
  }) {
    return deleteObject(name: name);
  }

  Future<void> deleteById({
    required String id,
  }) {
    return deleteObject(id: id);
  }

  Future<void> delete_by_id({
    required String id,
  }) {
    return deleteById(id: id);
  }

  Future<void> moveInsideBbox({
    required String target,
    required Offset newOrigin,
  }) async {
    await _onMoveObject(
      target: target,
      newOrigin: newOrigin,
    );
  }

  Future<void> move_inside_bbox({
    required String target,
    required Offset newOrigin,
  }) {
    return moveInsideBbox(
      target: target,
      newOrigin: newOrigin,
    );
  }

  Future<void> linkToImage({
    required String target,
    required String imageName,
  }) async {
    await _onLinkToImage(
      target: target,
      imageName: imageName,
    );
  }

  Future<void> link_to_image({
    required String target,
    required String imageName,
  }) {
    return linkToImage(
      target: target,
      imageName: imageName,
    );
  }

  Future<void> deleteSelf({
    required String target,
  }) async {
    await _onDeleteSelf(target: target);
  }

  Future<void> delete_self({
    required String target,
  }) {
    return deleteSelf(target: target);
  }

  Future<void> drawCluster({
    required String clusterRef,
  }) async {
    await _onDrawCluster(clusterRef: clusterRef);
  }

  Future<void> draw_cluster({
    required String clusterRef,
  }) {
    return drawCluster(clusterRef: clusterRef);
  }

  Future<void> highlightCluster({
    required String clusterRef,
  }) async {
    await _onHighlightCluster(clusterRef: clusterRef);
  }

  Future<void> highlight_cluster({
    required String clusterRef,
  }) {
    return highlightCluster(clusterRef: clusterRef);
  }

  Future<void> zoomCluster({
    required String clusterRef,
  }) async {
    await _onZoomCluster(clusterRef: clusterRef);
  }

  Future<void> zoom_cluster({
    required String clusterRef,
  }) {
    return zoomCluster(clusterRef: clusterRef);
  }

  Future<void> writeLabel({
    required String clusterRef,
    required String text,
  }) async {
    await _onWriteLabel(
      clusterRef: clusterRef,
      text: text,
    );
  }

  Future<void> write_label({
    required String clusterRef,
    required String text,
  }) {
    return writeLabel(
      clusterRef: clusterRef,
      text: text,
    );
  }

  Future<void> connectClusterToCluster({
    required String fromClusterRef,
    required String toClusterRef,
  }) async {
    await _onConnectClusters(
      fromClusterRef: fromClusterRef,
      toClusterRef: toClusterRef,
    );
  }

  Future<void> connect_cluster_to_cluster({
    required String fromClusterRef,
    required String toClusterRef,
  }) {
    return connectClusterToCluster(
      fromClusterRef: fromClusterRef,
      toClusterRef: toClusterRef,
    );
  }
}

class WhiteboardActionHost {
  WhiteboardActionHost._();

  static final WhiteboardActionHost instance = WhiteboardActionHost._();

  WhiteboardActions? _boundActions;

  bool get isBound => _boundActions != null;

  void bind(WhiteboardActions actions) {
    _boundActions = actions;
  }

  void unbind(WhiteboardActions actions) {
    if (identical(_boundActions, actions)) {
      _boundActions = null;
    }
  }

  WhiteboardActions get _actions {
    final actions = _boundActions;
    if (actions == null) {
      throw StateError(
        'Whiteboard actions are not bound. Launch the whiteboard first.',
      );
    }
    return actions;
  }

  Future<void> drawImage({
    required String fileName,
    required Offset origin,
    required double objectScale,
    String? boardObjectId,
    String? logicalName,
    String? processedId,
    Set<String>? aliases,
  }) {
    return _actions.drawImage(
      fileName: fileName,
      origin: origin,
      objectScale: objectScale,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      processedId: processedId,
      aliases: aliases,
    );
  }

  Future<void> draw_image({
    required String fileName,
    required Offset origin,
    required double objectScale,
    String? boardObjectId,
    String? logicalName,
    String? processedId,
    Set<String>? aliases,
  }) {
    return drawImage(
      fileName: fileName,
      origin: origin,
      objectScale: objectScale,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      processedId: processedId,
      aliases: aliases,
    );
  }

  Future<void> writeText({
    required String prompt,
    required Offset origin,
    required double letterSize,
    double? strokeSlowdown,
    String? boardObjectId,
    String? logicalName,
    String? attachedToObjectId,
    Set<String>? aliases,
  }) {
    return _actions.writeText(
      prompt: prompt,
      origin: origin,
      letterSize: letterSize,
      strokeSlowdown: strokeSlowdown,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      attachedToObjectId: attachedToObjectId,
      aliases: aliases,
    );
  }

  Future<void> write_text({
    required String prompt,
    required Offset origin,
    required double letterSize,
    double? strokeSlowdown,
    String? boardObjectId,
    String? logicalName,
    String? attachedToObjectId,
    Set<String>? aliases,
  }) {
    return writeText(
      prompt: prompt,
      origin: origin,
      letterSize: letterSize,
      strokeSlowdown: strokeSlowdown,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      attachedToObjectId: attachedToObjectId,
      aliases: aliases,
    );
  }

  Future<void> deleteObject({
    String? id,
    String? name,
  }) {
    return _actions.deleteObject(id: id, name: name);
  }

  Future<void> delete_object({
    String? id,
    String? name,
  }) {
    return deleteObject(id: id, name: name);
  }

  Future<void> deleteImage({
    required String name,
  }) {
    return _actions.deleteObject(name: name);
  }

  Future<void> delete_image({
    required String name,
  }) {
    return deleteImage(name: name);
  }

  Future<void> deleteById({
    required String id,
  }) {
    return _actions.deleteObject(id: id);
  }

  Future<void> delete_by_id({
    required String id,
  }) {
    return deleteById(id: id);
  }

  Future<void> moveInsideBbox({
    required String target,
    required Offset newOrigin,
  }) {
    return _actions.moveInsideBbox(
      target: target,
      newOrigin: newOrigin,
    );
  }

  Future<void> move_inside_bbox({
    required String target,
    required Offset newOrigin,
  }) {
    return moveInsideBbox(
      target: target,
      newOrigin: newOrigin,
    );
  }

  Future<void> linkToImage({
    required String target,
    required String imageName,
  }) {
    return _actions.linkToImage(
      target: target,
      imageName: imageName,
    );
  }

  Future<void> link_to_image({
    required String target,
    required String imageName,
  }) {
    return linkToImage(
      target: target,
      imageName: imageName,
    );
  }

  Future<void> deleteSelf({
    required String target,
  }) {
    return _actions.deleteSelf(target: target);
  }

  Future<void> delete_self({
    required String target,
  }) {
    return deleteSelf(target: target);
  }

  Future<void> drawCluster({
    required String clusterRef,
  }) {
    return _actions.drawCluster(clusterRef: clusterRef);
  }

  Future<void> draw_cluster({
    required String clusterRef,
  }) {
    return drawCluster(clusterRef: clusterRef);
  }

  Future<void> highlightCluster({
    required String clusterRef,
  }) {
    return _actions.highlightCluster(clusterRef: clusterRef);
  }

  Future<void> highlight_cluster({
    required String clusterRef,
  }) {
    return highlightCluster(clusterRef: clusterRef);
  }

  Future<void> zoomCluster({
    required String clusterRef,
  }) {
    return _actions.zoomCluster(clusterRef: clusterRef);
  }

  Future<void> zoom_cluster({
    required String clusterRef,
  }) {
    return zoomCluster(clusterRef: clusterRef);
  }

  Future<void> writeLabel({
    required String clusterRef,
    required String text,
  }) {
    return _actions.writeLabel(
      clusterRef: clusterRef,
      text: text,
    );
  }

  Future<void> write_label({
    required String clusterRef,
    required String text,
  }) {
    return writeLabel(
      clusterRef: clusterRef,
      text: text,
    );
  }

  Future<void> connectClusterToCluster({
    required String fromClusterRef,
    required String toClusterRef,
  }) {
    return _actions.connectClusterToCluster(
      fromClusterRef: fromClusterRef,
      toClusterRef: toClusterRef,
    );
  }

  Future<void> connect_cluster_to_cluster({
    required String fromClusterRef,
    required String toClusterRef,
  }) {
    return connectClusterToCluster(
      fromClusterRef: fromClusterRef,
      toClusterRef: toClusterRef,
    );
  }
}
