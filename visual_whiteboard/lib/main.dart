import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'lesson_tts_stream_accepter.dart';
import 'whiteboard_actions.dart';

void main() {
  runApp(const WhiteboardApp());
}

class WhiteboardApp extends StatelessWidget {
  const WhiteboardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: VectorViewerScreen(),
    );
  }
}

class VectorViewerScreen extends StatefulWidget {
  const VectorViewerScreen({super.key});

  @override
  State<VectorViewerScreen> createState() => _VectorViewerScreenState();
}

class _VectorViewerScreenState extends State<VectorViewerScreen>
    with SingleTickerProviderStateMixin {
  static String get _vectorsFolder => _resolveBackendSubdir('StrokeVectors');

  static String get _fontVectorsFolder => _resolveBackendSubdir('Font');

  static String get _fontMetricsPath => '${_fontVectorsFolder}\\font_metrics.json';

  static String get _clustersFolder =>
      _resolveBackendSubdir(r'PipelineOutputs\_diagram_part_stroke_maps');

  static const String _lessonWsUrl = 'ws://127.0.0.1:8765';

  static const double _targetResolution = 2000.0;
  static const double _defaultImageObjectScale = 0.75;
  static const double _basePenWidthPx = 4.0;
  static const double _boardWidth = _targetResolution;
  static const double _boardHeight = _targetResolution;
  static const int _maxDisplayPointsPerStroke = 120;
  static const bool _forceMaxObjectAnimationDuration = true;
  static const double _forcedMaxObjectAnimationDurationSec = 5.0;
  static const bool _enableParallelStrokeWorkers = true;
  static const int _maxParallelStrokeWorkers = 24;

  static const String _apiBaseUrl = 'http://127.0.0.1:8000';
  static const bool _backendEnabled = true;

  Uri _apiUri(String path) => Uri.parse('$_apiBaseUrl$path');

  List<StrokePolyline> _polyStrokes = const [];
  List<StrokeCubic> _cubicStrokes = const [];

  List<DrawableStroke> _drawableStrokes = const [];
  List<DrawableStroke> _staticStrokes = const [];
  List<DrawableStroke> _animStrokes = const [];

  double? _srcWidth;
  double? _srcHeight;

  String _status = 'Idle';

  late final AnimationController _controller;
  double _animValue = 0.0;

  bool _stepMode = false;
  int _stepStrokeCount = 0;

  double _minStrokeTimeSec = 0.17;
  double _maxStrokeTimeSec = 0.8;
  double _lengthTimePerKPxSec = 0.3;
  double _curvatureExtraMaxSec = 0.08;
  double _curvatureProfileFactor = 1.5;
  double _curvatureAngleScale = 80.0;
  double _baseTravelTimeSec = 0.15;
  double _travelTimePerKPxSec = 0.12;
  double _minTravelTimeSec = 0.15;
  double _maxTravelTimeSec = 0.35;
  double _globalSpeedMultiplier = 1.0;
  double _textLetterGapPx = 20.0;
  double _strokeSpeedStartPct = 0.08;
  double _strokeSpeedEndPct = 0.25;
  double _strokeSpeedPeakMult = 2.50;
  double _strokeSpeedPeakTime = 0.6;

  final TextEditingController _fileNameController =
      TextEditingController(text: 'processed_4.json');
  final TextEditingController _posXController =
      TextEditingController(text: '0');
  final TextEditingController _posYController =
      TextEditingController(text: '0');
  final TextEditingController _scaleController =
      TextEditingController(text: _defaultImageObjectScale.toString());

  final List<String> _drawnJsonNames = [];
  String? _selectedEraseName;

  final Map<int, GlyphData> _glyphCache = {};
  double? _fontLineHeightPx;
  double? _fontImageHeightPx;

  bool _animIsText = false;
  static const double _defaultTextStrokeSlowdown = 8.0;
  double _textStrokeSlowdown = _defaultTextStrokeSlowdown;
  double _textStrokeBaseTimeSec = 0.017;
  double _textStrokeCurveExtraFrac = 0.25;
  double _textLetterPauseSec = 0.0;
  double _textBaseFontSizeRef = 200.0;

  final TextEditingController _textPromptController =
      TextEditingController(text: 'Hello, world');
  final TextEditingController _textXController =
      TextEditingController(text: '0');
  final TextEditingController _textYController =
      TextEditingController(text: '0');
  final TextEditingController _textSizeController =
      TextEditingController(text: '180');
  final TextEditingController _textGapController =
      TextEditingController(text: '20');

  final Map<String, BoardObjectRecord> _boardObjectsById = {};
  final Map<String, Set<String>> _boardObjectIdsByAlias = {};
  final Map<String, DiagramRuntimeState> _diagramStatesByObjectId = {};
  final Set<String> _highlightedClusterRefs = <String>{};
  final Map<String, DiagramPlacedLabel> _diagramLabelsByClusterRef = {};
  final List<DiagramConnectionRecord> _diagramConnections = [];
  String? _zoomedClusterRef;

  static String get _processedImagesFolder =>
      _resolveBackendSubdir('ProcessedImages');

  final Map<String, double> _objectSaturationById = {};
  final Map<String, Set<String>> _activeHighlightRefsByDiagramId = {};
  final Map<String, int> _highlightRefCountsByClusterRef = {};
  final Map<String, double> _diagramNonFocusSaturationByObjectId = {};
  final Map<String, VectorBlueprint> _vectorBlueprintCache = {};
  final Map<String, List<DiagramLabelAnchor>> _labelAnchorCacheByProcessedId = {};
  final Map<int, ActiveLessonAction> _activeLessonActionsByGlobalIndex = {};
  final Set<String> _processedSilenceEventKeys = <String>{};
  Future<void> _lessonActionQueue = Future<void>.value();
  String? _zoomedDiagramObjectId;
  double _boardZoomScale = 1.0;
  Offset _boardZoomWorldCenter = Offset.zero;
  int _tempObjectCounter = 0;
  double _activeAnimationDurationSec = 0.0;
  int _activeAnimationWorkerCount = 1;
  Completer<void>? _animationCompletionCompleter;

  late final WhiteboardActions _whiteboardActions;
  late final LessonTtsStreamAccepter _lessonStreamAccepter;

  @override
  void initState() {
    super.initState();
    _whiteboardActions = WhiteboardActions(
      onDrawImage: _addObjectFromJson,
      onWriteText: _writeTextPrompt,
      onDeleteObject: ({String? id, String? name}) =>
          _deleteObject(id: id, name: name),
      onMoveObject: ({required String target, required Offset newOrigin}) =>
          _moveObjectTo(target: target, newOrigin: newOrigin),
      onLinkToImage: ({required String target, required String imageName}) =>
          _linkObjectToImage(target: target, imageName: imageName),
      onDeleteSelf: ({required String target}) => _deleteObject(name: target),
      onDrawCluster: ({required String clusterRef}) =>
          _drawCluster(clusterRef: clusterRef),
      onHighlightCluster: ({required String clusterRef}) =>
          _highlightCluster(clusterRef: clusterRef),
      onZoomCluster: ({required String clusterRef}) =>
          _zoomCluster(clusterRef: clusterRef),
      onWriteLabel: ({required String clusterRef, required String text}) =>
          _writeDiagramLabel(clusterRef: clusterRef, text: text),
      onConnectClusters: ({
        required String fromClusterRef,
        required String toClusterRef,
      }) =>
          _connectClusters(
        fromClusterRef: fromClusterRef,
        toClusterRef: toClusterRef,
      ),
    );
    WhiteboardActionHost.instance.bind(_whiteboardActions);
    _lessonStreamAccepter = LessonTtsStreamAccepter(
      websocketUrl: _lessonWsUrl,
      autoReconnect: true,
      onPacket: _handleLessonStreamPacket,
      onError: _handleLessonStreamError,
    );

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 20000),
    )
      ..addListener(() {
        if (!_stepMode) {
          setState(() {
            _animValue = _controller.value;
          });
        }
      })
      ..addStatusListener((status) {
        if (status == AnimationStatus.completed) {
          if (_animStrokes.isNotEmpty) {
            setState(() {
              _staticStrokes = [..._staticStrokes, ..._animStrokes];
              _animStrokes = const [];
              _drawableStrokes = [..._staticStrokes];
              _animValue = 1.0;
              _status =
                  'Animation finished. Total strokes: ${_drawableStrokes.length}';
            });
          }
          final completer = _animationCompletionCompleter;
          _animationCompletionCompleter = null;
          if (completer != null && !completer.isCompleted) {
            completer.complete();
          }
        }
      });

    _loadObjectsFromBackend();
    unawaited(_lessonStreamAccepter.connect());
  }

  @override
  void dispose() {
    WhiteboardActionHost.instance.unbind(_whiteboardActions);
    unawaited(_lessonStreamAccepter.dispose());
    _controller.dispose();
    _fileNameController.dispose();
    _posXController.dispose();
    _posYController.dispose();
    _scaleController.dispose();
    _textPromptController.dispose();
    _textXController.dispose();
    _textYController.dispose();
    _textSizeController.dispose();
    _textGapController.dispose();
    super.dispose();
  }

  void _handleLessonStreamPacket(LessonStreamPacket packet) {
    if (!mounted) {
      return;
    }

    if (packet.rawType == 'deletion_silence') {
      final emitPhase = (packet.rawJson['emit_phase'] ?? '').toString();
      final chapterIndex = _coerceInt(packet.rawJson['chapter_index']);

      setState(() {
        _status =
            'Received deletion silence packet ($emitPhase) for chapter $chapterIndex.';
      });

      _enqueueLessonOperation(() async {
        if (emitPhase == 'start') {
          await _executeDeletionSilenceFromRaw(packet.rawJson);
        }
      });
      return;
    }

    final actionPacket = packet.actionPacket;
    if (actionPacket != null) {
      setState(() {
        _status =
            'Received ${actionPacket.action.type} packet (${actionPacket.emitPhase}) '
            'for chapter ${actionPacket.chapterIndex}.';
      });

      _enqueueLessonOperation(() async {
        if (actionPacket.isStartPhase) {
          await _executeActionPacket(actionPacket);
        } else {
          await _executeActionPacketEnd(actionPacket);
        }
      });
      return;
    }

    setState(() {
      _status = 'Received lesson stream packet: ${packet.rawType}.';
    });
  }

  void _handleLessonStreamError(Object error, StackTrace stackTrace) {
    if (!mounted) {
      return;
    }

    setState(() {
      _status = 'Lesson stream error: $error';
    });
  }

  void _enqueueLessonOperation(Future<void> Function() operation) {
    _lessonActionQueue = _lessonActionQueue.then((_) => operation()).catchError(
      (Object error, StackTrace stackTrace) {
        _handleLessonStreamError(error, stackTrace);
      },
    );
  }

  int _coerceInt(dynamic value, [int fallback = 0]) {
    if (value is int) {
      return value;
    }
    if (value is num) {
      return value.toInt();
    }
    return int.tryParse(value?.toString() ?? '') ?? fallback;
  }

  Future<void> _executeDeletionSilenceFromRaw(
    Map<String, dynamic> raw,
  ) async {
    final eventKey =
        '${_coerceInt(raw['chapter_index'])}:${_coerceInt(raw['event_index'])}';
    if (_processedSilenceEventKeys.contains(eventKey)) {
      return;
    }
    _processedSilenceEventKeys.add(eventKey);

    final protectedNames = <String>{};
    final blocked = raw['delete_targets_blocked_due_active_diagram'];
    if (blocked is List) {
      protectedNames.addAll(
        blocked.map((e) => e.toString().trim()).where((e) => e.isNotEmpty),
      );
    }
    final shifted = raw['manual_shift_targets_due_active_diagram'];
    if (shifted is List) {
      protectedNames.addAll(
        shifted.map((e) => e.toString().trim()).where((e) => e.isNotEmpty),
      );
    }

    final protectedRootIds = <String>{};
    for (final name in protectedNames) {
      protectedRootIds.addAll(_resolveObjectIds(name));
    }

    final candidates = _boardObjectsById.values
        .where((record) {
          if (protectedRootIds.contains(record.objectId)) {
            return false;
          }

          final ownerDiagramId = record.ownerDiagramObjectId;
          if (ownerDiagramId != null &&
              protectedRootIds.contains(ownerDiagramId)) {
            return false;
          }

          return true;
        })
        .toList(growable: false)
      ..sort((a, b) {
        final aBounds = _objectBounds(a.objectId);
        final bBounds = _objectBounds(b.objectId);

        final aTop = aBounds?.top ?? a.origin.dy;
        final bTop = bBounds?.top ?? b.origin.dy;
        final byTop = aTop.compareTo(bTop);
        if (byTop != 0) {
          return byTop;
        }

        final aLeft = aBounds?.left ?? a.origin.dx;
        final bLeft = bBounds?.left ?? b.origin.dx;
        return aLeft.compareTo(bLeft);
      });

    if (candidates.isEmpty) {
      if (!mounted) {
        return;
      }
      setState(() {
        _status =
            'Deletion silence: nothing to delete. Protected roots: ${protectedRootIds.length}.';
      });
      return;
    }

    for (final record in candidates) {
      await _deleteObject(id: record.objectId, silentIfMissing: true);
      await Future<void>.delayed(const Duration(milliseconds: 35));
    }

    if (!mounted) {
      return;
    }
    setState(() {
      _status =
          'Deletion silence cleared ${candidates.length} object(s). Protected roots: ${protectedRootIds.length}.';
    });
  }

  Future<void> _executeActionPacket(LessonActionPacket packet) async {
    final action = packet.action;

    if (packet.eventKind == 'silence' &&
        (action.type == 'delete_by_name' || action.type == 'delete_by_id')) {
      await _executeDeletionSilenceFromRaw(packet.rawJson);
      return;
    }

    switch (action.type) {
      case 'draw_image':
        final fileName = _resolveVectorFileName(packet, action);
        if (fileName == null) {
          setState(() {
            _status =
                'draw_image skipped: no processed_id/file name for ${packet.name}';
          });
          return;
        }

        final logicalName = action.target?.trim().isNotEmpty == true
            ? action.target!.trim()
            : packet.name;
        final aliases = <String>{
          if (logicalName.isNotEmpty) logicalName,
          if (packet.name.isNotEmpty) packet.name,
          if (packet.processedId.isNotEmpty) packet.processedId,
          fileName,
          _stemWithoutJson(fileName),
        };

        await _whiteboardActions.drawImage(
          fileName: fileName,
          origin: Offset(action.x ?? 0.0, action.y ?? 0.0),
          objectScale: action.scale ?? _defaultImageObjectScale,
          boardObjectId: packet.processedId.isNotEmpty
              ? packet.processedId
              : _stemWithoutJson(fileName),
          logicalName: logicalName,
          processedId: packet.processedId.isEmpty ? null : packet.processedId,
          aliases: aliases,
        );
        await _waitForStrokeAnimationComplete();
        return;

      case 'write_text':
        final prompt = action.text ?? action.target ?? packet.name;
        if (prompt.trim().isEmpty) {
          return;
        }

        final attachedIds = _resolveObjectIds(action.target);
        final attachedToObjectId = attachedIds.isEmpty ? null : attachedIds.first;

        await _whiteboardActions.writeText(
          prompt: prompt,
          origin: Offset(action.x ?? 0.0, action.y ?? 0.0),
          letterSize: _deriveLetterSizeFromScale(action.scale ?? 1.0),
          boardObjectId: 'text_action_${packet.globalActionIndex}',
          logicalName: prompt,
          attachedToObjectId: attachedToObjectId,
          aliases: <String>{
            prompt,
            if ((action.target ?? '').trim().isNotEmpty) action.target!.trim(),
            if (packet.name.trim().isNotEmpty) packet.name.trim(),
          },
        );
        await _waitForStrokeAnimationComplete();
        return;

      case 'move_inside_bbox':
        await _moveObjectTo(
          target: action.target ?? packet.name,
          newOrigin: Offset(action.newX ?? 0.0, action.newY ?? 0.0),
        );
        return;

      case 'link_to_image':
        if ((action.target ?? '').isEmpty || (action.imageName ?? '').isEmpty) {
          return;
        }
        await _startLinkToImageAction(packet, action);
        return;

      case 'delete_self':
        await _whiteboardActions.deleteSelf(
          target: _bestDeleteTarget(packet, action),
        );
        return;

      case 'delete_by_name':
        final target = action.target?.trim();
        if (target == null || target.isEmpty) {
          return;
        }
        await _whiteboardActions.deleteObject(name: target);
        return;

      case 'delete_by_id':
        final targetId = action.imageId?.trim();
        if (targetId == null || targetId.isEmpty) {
          return;
        }
        await _whiteboardActions.deleteObject(id: targetId);
        return;

      case 'draw_cluster':
        final clusterRef = action.clusterName?.trim();
        if (clusterRef == null || clusterRef.isEmpty) {
          return;
        }
        await _startDrawClusterAction(packet, clusterRef);
        return;

      case 'highlight_cluster':
        final clusterRef = action.clusterName?.trim();
        if (clusterRef == null || clusterRef.isEmpty) {
          return;
        }
        await _startHighlightClusterAction(packet, clusterRef);
        return;

      case 'zoom_cluster':
        final clusterRef = action.clusterName?.trim();
        if (clusterRef == null || clusterRef.isEmpty) {
          return;
        }
        await _startZoomClusterAction(packet, clusterRef);
        return;

      case 'write_label':
        final clusterRef = action.clusterName?.trim();
        final labelText = action.text?.trim();
        if (clusterRef == null || clusterRef.isEmpty || labelText == null || labelText.isEmpty) {
          return;
        }
        await _startWriteLabelAction(packet, clusterRef, labelText);
        return;

      case 'connect_cluster_to_cluster':
        final fromCluster = action.fromCluster?.trim();
        final toCluster = action.toCluster?.trim();
        if (fromCluster == null ||
            fromCluster.isEmpty ||
            toCluster == null ||
            toCluster.isEmpty) {
          return;
        }
        await _startConnectClustersAction(packet, fromCluster, toCluster);
        return;

      default:
        setState(() {
          _status = 'Unsupported action type: ${action.type}';
        });
    }
  }

  Future<void> _executeActionPacketEnd(LessonActionPacket packet) async {
    switch (packet.action.type) {
      case 'link_to_image':
      case 'highlight_cluster':
      case 'zoom_cluster':
      case 'connect_cluster_to_cluster':
        await _runCounterActionByIndex(packet.globalActionIndex);
        return;
      default:
        return;
    }
  }

  Future<void> _runCounterActionByIndex(int globalActionIndex) async {
    final action = _activeLessonActionsByGlobalIndex[globalActionIndex];
    if (action == null) {
      return;
    }
    await _runCounterAction(action);
  }

  Future<void> _runCounterAction(ActiveLessonAction action) async {
    switch (action.type) {
      case 'link_to_image':
        if (action.tempObjectId != null) {
          await _deleteObject(id: action.tempObjectId, silentIfMissing: true);
        }
        break;
      case 'highlight_cluster':
        if (action.primaryClusterRef != null && action.diagramObjectId != null) {
          await _removeHighlightClusterRef(
            diagramObjectId: action.diagramObjectId!,
            clusterRef: action.primaryClusterRef!,
          );
        }
        break;
      case 'zoom_cluster':
        if (action.primaryClusterRef != null && action.diagramObjectId != null) {
          await _deactivateZoom(
            diagramObjectId: action.diagramObjectId!,
            clusterRef: action.primaryClusterRef!,
          );
        }
        break;
      case 'connect_cluster_to_cluster':
        if (action.tempObjectId != null) {
          await _deleteObject(id: action.tempObjectId, silentIfMissing: true);
        }
        if (action.diagramObjectId != null && action.primaryClusterRef != null) {
          await _removeHighlightClusterRef(
            diagramObjectId: action.diagramObjectId!,
            clusterRef: action.primaryClusterRef!,
          );
        }
        if (action.secondaryDiagramObjectId != null &&
            action.secondaryClusterRef != null) {
          await _removeHighlightClusterRef(
            diagramObjectId: action.secondaryDiagramObjectId!,
            clusterRef: action.secondaryClusterRef!,
          );
        }
        break;
    }
    _activeLessonActionsByGlobalIndex.remove(action.globalActionIndex);
  }

  Future<void> _cancelIncompatibleDiagramActionsForStart({
    required String incomingType,
    required Set<String> incomingDiagramIds,
    required int incomingActionId,
  }) async {
    final activeActions = _activeLessonActionsByGlobalIndex.values.toList();
    for (final active in activeActions) {
      if (!active.isDiagramAction || active.globalActionIndex == incomingActionId) {
        continue;
      }

      final sameDiagramHighlight = incomingType == 'highlight_cluster' &&
          active.type == 'highlight_cluster' &&
          incomingDiagramIds.length == 1 &&
          active.affectedDiagramObjectIds.length == 1 &&
          incomingDiagramIds.first == active.affectedDiagramObjectIds.first;

      if (sameDiagramHighlight) {
        continue;
      }

      await _runCounterAction(active);
    }
  }

  Future<void> _handleSilenceDeletionStart(LessonActionPacket packet) {
    return _executeDeletionSilenceFromRaw(packet.rawJson);
  }

  Future<void> _startLinkToImageAction(
    LessonActionPacket packet,
    LessonBoardAction action,
  ) async {
    final targetIds = _resolveObjectIds(action.target);
    final imageIds = _resolveObjectIds(action.imageName);
    if (targetIds.isEmpty || imageIds.isEmpty) {
      setState(() {
        _status =
            'link_to_image skipped: target="${action.target}", image="${action.imageName}"';
      });
      return;
    }

    final fromId = targetIds.first;
    final toId = imageIds.first;
    final fromBounds = _objectBounds(fromId);
    final toBounds = _objectBounds(toId);
    if (fromBounds == null || toBounds == null) {
      return;
    }

    final segments = _buildDoubleArrowConnectorSegments(
      fromBounds: fromBounds,
      toBounds: toBounds,
      gap: 30.0,
      arrowLength: 48.0,
      arrowHalfWidth: 18.0,
    );

    final connectorId = _nextTempObjectId('link');
    await _addGeneratedPolylineObject(
      objectId: connectorId,
      displayName: connectorId,
      polylines: segments,
      color: Colors.black,
      isTemporary: true,
      ownerDiagramObjectId: null,
      syncWithBackend: false,
      aliases: <String>{connectorId, 'connector_${packet.globalActionIndex}'},
      awaitAnimation: true,
    );

    _activeLessonActionsByGlobalIndex[packet.globalActionIndex] = ActiveLessonAction(
      globalActionIndex: packet.globalActionIndex,
      type: 'link_to_image',
      tempObjectId: connectorId,
      affectedDiagramObjectIds: const <String>{},
    );
  }

  Future<void> _startDrawClusterAction(
    LessonActionPacket packet,
    String clusterRef,
  ) async {
    final resolved = await _resolveOrCreateCluster(packet, clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'draw_cluster skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    await _cancelIncompatibleDiagramActionsForStart(
      incomingType: 'draw_cluster',
      incomingDiagramIds: <String>{resolved.diagram.objectId},
      incomingActionId: packet.globalActionIndex,
    );

    await _drawCluster(clusterRef: clusterRef, resolved: resolved);
  }

  Future<void> _startHighlightClusterAction(
    LessonActionPacket packet,
    String clusterRef,
  ) async {
    final resolved = await _resolveOrCreateCluster(packet, clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'highlight_cluster skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    await _cancelIncompatibleDiagramActionsForStart(
      incomingType: 'highlight_cluster',
      incomingDiagramIds: <String>{resolved.diagram.objectId},
      incomingActionId: packet.globalActionIndex,
    );

    await _highlightCluster(clusterRef: clusterRef, resolved: resolved);
    _activeLessonActionsByGlobalIndex[packet.globalActionIndex] = ActiveLessonAction(
      globalActionIndex: packet.globalActionIndex,
      type: 'highlight_cluster',
      diagramObjectId: resolved.diagram.objectId,
      primaryClusterRef: resolved.cluster.clusterRef,
      affectedDiagramObjectIds: <String>{resolved.diagram.objectId},
    );
  }

  Future<void> _startZoomClusterAction(
    LessonActionPacket packet,
    String clusterRef,
  ) async {
    final resolved = await _resolveOrCreateCluster(packet, clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'zoom_cluster skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    await _cancelIncompatibleDiagramActionsForStart(
      incomingType: 'zoom_cluster',
      incomingDiagramIds: <String>{resolved.diagram.objectId},
      incomingActionId: packet.globalActionIndex,
    );

    await _zoomCluster(clusterRef: clusterRef, resolved: resolved);
    _activeLessonActionsByGlobalIndex[packet.globalActionIndex] = ActiveLessonAction(
      globalActionIndex: packet.globalActionIndex,
      type: 'zoom_cluster',
      diagramObjectId: resolved.diagram.objectId,
      primaryClusterRef: resolved.cluster.clusterRef,
      affectedDiagramObjectIds: <String>{resolved.diagram.objectId},
    );
  }

  Future<void> _startWriteLabelAction(
    LessonActionPacket packet,
    String clusterRef,
    String labelText,
  ) async {
    final resolved = await _resolveOrCreateCluster(packet, clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'write_label skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    await _cancelIncompatibleDiagramActionsForStart(
      incomingType: 'write_label',
      incomingDiagramIds: <String>{resolved.diagram.objectId},
      incomingActionId: packet.globalActionIndex,
    );

    await _writeDiagramLabel(
      clusterRef: clusterRef,
      text: labelText,
      resolved: resolved,
      actionKey: packet.globalActionIndex,
    );
  }

  Future<void> _startConnectClustersAction(
    LessonActionPacket packet,
    String fromClusterRef,
    String toClusterRef,
  ) async {
    final fromResolved = await _resolveOrCreateCluster(packet, fromClusterRef);
    final toResolved = await _resolveOrCreateCluster(packet, toClusterRef);
    if (fromResolved == null || toResolved == null) {
      setState(() {
        _status =
            'connect_cluster_to_cluster skipped: "$fromClusterRef" -> "$toClusterRef".';
      });
      return;
    }

    await _cancelIncompatibleDiagramActionsForStart(
      incomingType: 'connect_cluster_to_cluster',
      incomingDiagramIds: <String>{
        fromResolved.diagram.objectId,
        toResolved.diagram.objectId,
      },
      incomingActionId: packet.globalActionIndex,
    );

    await _applyHighlightClusterRef(
      diagramObjectId: fromResolved.diagram.objectId,
      clusterRef: fromResolved.cluster.clusterRef,
    );
    await _applyHighlightClusterRef(
      diagramObjectId: toResolved.diagram.objectId,
      clusterRef: toResolved.cluster.clusterRef,
    );

    final fromCenter = _clusterCenterForResolved(fromResolved) ??
        (_objectBounds(fromResolved.diagram.objectId)?.center);
    final toCenter = _clusterCenterForResolved(toResolved) ??
        (_objectBounds(toResolved.diagram.objectId)?.center);
    if (fromCenter == null || toCenter == null) {
      return;
    }

    final dottedSegments = _buildDottedLineSegments(
      from: fromCenter,
      to: toCenter,
      gapFromStart: 24.0,
      gapFromEnd: 24.0,
      dashLength: 22.0,
      dashGap: 16.0,
    );

    final connectorId = _nextTempObjectId('diagram_connect');
    await _addGeneratedPolylineObject(
      objectId: connectorId,
      displayName: connectorId,
      polylines: dottedSegments,
      color: const Color(0xFF26A69A),
      isTemporary: true,
      ownerDiagramObjectId: fromResolved.diagram.objectId,
      syncWithBackend: false,
      aliases: <String>{connectorId, 'diagram_connect_${packet.globalActionIndex}'},
      awaitAnimation: true,
    );

    _activeLessonActionsByGlobalIndex[packet.globalActionIndex] = ActiveLessonAction(
      globalActionIndex: packet.globalActionIndex,
      type: 'connect_cluster_to_cluster',
      diagramObjectId: fromResolved.diagram.objectId,
      primaryClusterRef: fromResolved.cluster.clusterRef,
      secondaryDiagramObjectId: toResolved.diagram.objectId,
      secondaryClusterRef: toResolved.cluster.clusterRef,
      tempObjectId: connectorId,
      affectedDiagramObjectIds: <String>{
        fromResolved.diagram.objectId,
        toResolved.diagram.objectId,
      },
    );
  }

  Future<ResolvedDiagramCluster?> _resolveOrCreateCluster(
    LessonActionPacket packet,
    String rawClusterRef,
  ) async {
    var resolved = _resolveCluster(rawClusterRef);
    if (resolved != null) {
      return resolved;
    }

    final parsed = DiagramClusterTarget.tryParse(rawClusterRef);
    if (parsed == null) {
      return null;
    }

    final processedId = packet.processedId.trim();
    if (processedId.isEmpty) {
      return null;
    }

    final objectId = processedId;
    final existing = _boardObjectsById[objectId];
    if (existing == null) {
      final blueprint = await _loadVectorBlueprint('${processedId}.json');
      if (blueprint == null) {
        return null;
      }
      final origin = Offset(packet.action.x ?? 0.0, packet.action.y ?? 0.0);
      final scale = packet.action.scale ?? _defaultImageObjectScale;
      final record = BoardObjectRecord(
        objectId: objectId,
        kind: BoardObjectKind.image,
        displayName: parsed.diagramName,
        origin: origin,
        scale: scale,
        fileName: blueprint.fileName,
        processedId: processedId,
        aliases: <String>{
          parsed.diagramName,
          packet.name,
          processedId,
          blueprint.fileName,
          _stemWithoutJson(blueprint.fileName),
        },
        sourceWidth: blueprint.srcWidth,
        sourceHeight: blueprint.srcHeight,
        sourceBounds: blueprint.sourceBounds,
      );
      setState(() {
        _registerObject(record);
      });
      await _loadDiagramStateIfPresent(
        objectId: objectId,
        diagramName: parsed.diagramName,
        processedId: processedId,
      );
    }

    resolved = _resolveCluster(rawClusterRef);
    return resolved;
  }

  String? _resolveVectorFileName(
    LessonActionPacket packet,
    LessonBoardAction action,
  ) {
    if (packet.processedId.trim().isNotEmpty) {
      return '${packet.processedId.trim()}.json';
    }

    final imageName = packet.imageName.trim();
    if (imageName.isNotEmpty) {
      return imageName.endsWith('.json') ? imageName : '$imageName.json';
    }

    final target = (action.target ?? '').trim();
    if (target.endsWith('.json')) {
      return target;
    }

    return null;
  }

  String _bestDeleteTarget(LessonActionPacket packet, LessonBoardAction action) {
    final rawTarget = (action.target ?? '').trim();
    if (rawTarget.isNotEmpty) {
      return rawTarget;
    }
    if (packet.processedId.trim().isNotEmpty) {
      return packet.processedId.trim();
    }
    return packet.name.trim();
  }

  double _deriveLetterSizeFromScale(double scale) {
    final safeScale = scale <= 0 ? 1.0 : scale;
    return _textBaseFontSizeRef * safeScale;
  }

  String _stemWithoutJson(String value) {
    if (value.toLowerCase().endsWith('.json')) {
      return value.substring(0, value.length - 5);
    }
    return value;
  }

  String _normalizeLookupKey(String value) {
    final trimmed = value.trim().toLowerCase();
    final collapsed = trimmed.replaceAll(RegExp(r'\s+'), ' ');
    return collapsed.replaceAll(RegExp(r'[^a-z0-9 ]'), '');
  }

  Iterable<String> _aliasLookupKeys(String value) sync* {
    final trimmed = value.trim();
    if (trimmed.isEmpty) {
      return;
    }
    yield trimmed;
    final normalized = _normalizeLookupKey(trimmed);
    if (normalized.isNotEmpty && normalized != trimmed) {
      yield normalized;
    }
  }

  void _registerObject(BoardObjectRecord record) {
    _boardObjectsById[record.objectId] = record;
    _objectSaturationById.putIfAbsent(record.objectId, () => 1.0);

    for (final alias in record.aliases) {
      for (final key in _aliasLookupKeys(alias)) {
        final bucket = _boardObjectIdsByAlias.putIfAbsent(key, () => <String>{});
        bucket.add(record.objectId);
      }
    }

    if (!record.isTemporary) {
      if (!_drawnJsonNames.contains(record.displayName)) {
        _drawnJsonNames.add(record.displayName);
      }
      _selectedEraseName ??= record.displayName;
    }
  }

  void _unregisterObject(String objectId) {
    final record = _boardObjectsById.remove(objectId);
    if (record == null) {
      return;
    }

    _objectSaturationById.remove(objectId);
    _diagramNonFocusSaturationByObjectId.remove(objectId);
    _activeHighlightRefsByDiagramId.remove(objectId);
    if (_zoomedDiagramObjectId == objectId) {
      _zoomedDiagramObjectId = null;
      _zoomedClusterRef = null;
      _boardZoomScale = 1.0;
      _boardZoomWorldCenter = Offset.zero;
    }

    for (final alias in record.aliases) {
      for (final key in _aliasLookupKeys(alias)) {
        final bucket = _boardObjectIdsByAlias[key];
        if (bucket == null) {
          continue;
        }
        bucket.remove(objectId);
        if (bucket.isEmpty) {
          _boardObjectIdsByAlias.remove(key);
        }
      }
    }

    final diagramState = _diagramStatesByObjectId.remove(objectId);
    if (diagramState != null) {
      for (final clusterRef in diagramState.clustersByCanonicalRef.keys) {
        _highlightedClusterRefs.remove(clusterRef);
        _highlightRefCountsByClusterRef.remove(clusterRef);
        _diagramLabelsByClusterRef.remove(clusterRef);
        if (_zoomedClusterRef == clusterRef) {
          _zoomedClusterRef = null;
        }
      }
    }

    _diagramConnections.removeWhere(
      (connection) =>
          connection.fromObjectId == objectId || connection.toObjectId == objectId,
    );
    _activeLessonActionsByGlobalIndex.removeWhere(
      (_, action) => action.tempObjectId == objectId,
    );

    if (!record.isTemporary) {
      final stillUsedDisplayName = _boardObjectsById.values
          .any((candidate) => !candidate.isTemporary && candidate.displayName == record.displayName);
      if (!stillUsedDisplayName) {
        _drawnJsonNames.remove(record.displayName);
      }

      if (_selectedEraseName == record.displayName) {
        _selectedEraseName = _drawnJsonNames.isEmpty ? null : _drawnJsonNames.last;
      }
    }
  }

  Set<String> _resolveObjectIds(String? rawKey) {
    final key = (rawKey ?? '').trim();
    if (key.isEmpty) {
      return <String>{};
    }

    final resolved = <String>{};

    if (_boardObjectsById.containsKey(key)) {
      resolved.add(key);
    }

    final fileStem = _stemWithoutJson(key);
    if (_boardObjectsById.containsKey(fileStem)) {
      resolved.add(fileStem);
    }

    for (final lookupKey in <String>{
      ..._aliasLookupKeys(key),
      ..._aliasLookupKeys(fileStem),
    }) {
      final bucket = _boardObjectIdsByAlias[lookupKey];
      if (bucket != null && bucket.isNotEmpty) {
        resolved.addAll(bucket);
      }
    }

    return resolved;
  }

  Rect? _objectBounds(String objectId) {
    final strokes = <DrawableStroke>[
      ..._staticStrokes.where((stroke) => stroke.jsonName == objectId),
      ..._animStrokes.where((stroke) => stroke.jsonName == objectId),
    ];
    if (strokes.isEmpty) {
      return null;
    }
    Rect bounds = strokes.first.bounds;
    for (int i = 1; i < strokes.length; i++) {
      bounds = bounds.expandToInclude(strokes[i].bounds);
    }
    return bounds;
  }

  Offset? _clusterCenterForResolved(ResolvedDiagramCluster resolved) {
    final bounds = _clusterBoundsForResolved(resolved);
    if (bounds == null) {
      return null;
    }
    return bounds.center;
  }

  Rect? _clusterBoundsForResolved(ResolvedDiagramCluster resolved) {
    final strokes = <DrawableStroke>[
      ..._staticStrokes,
      ..._animStrokes,
    ].where((stroke) {
      return stroke.jsonName == resolved.diagram.objectId &&
          resolved.cluster.strokeIndexes.contains(stroke.sourceStrokeIndex);
    }).toList(growable: false);
    if (strokes.isEmpty) {
      return null;
    }
    Rect bounds = strokes.first.bounds;
    for (int i = 1; i < strokes.length; i++) {
      bounds = bounds.expandToInclude(strokes[i].bounds);
    }
    return bounds;
  }

  Future<void> _deleteObject({
    String? id,
    String? name,
    bool silentIfMissing = false,
    bool animate = true,
  }) async {
    final ids = <String>{};
    if (id != null && id.trim().isNotEmpty) {
      ids.addAll(_resolveObjectIds(id));
      if (ids.isEmpty && _boardObjectsById.containsKey(id.trim())) {
        ids.add(id.trim());
      }
    }
    if (name != null && name.trim().isNotEmpty) {
      ids.addAll(_resolveObjectIds(name));
    }

    if (ids.isEmpty) {
      if (!silentIfMissing) {
        setState(() {
          _status = 'Delete skipped: no object matches id=$id name=$name';
        });
      }
      return;
    }

    if (animate) {
      for (final objectId in ids) {
        await _animateDeleteObject(objectId);
      }
    }

    final deletedRecords = <BoardObjectRecord>[];
    setState(() {
      _staticStrokes = _staticStrokes.where((s) => !ids.contains(s.jsonName)).toList();
      _animStrokes = _animStrokes.where((s) => !ids.contains(s.jsonName)).toList();
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];

      for (final objectId in ids) {
        final record = _boardObjectsById[objectId];
        if (record != null) {
          deletedRecords.add(record);
        }
        _unregisterObject(objectId);
      }

      _status =
          'Deleted ${ids.length} object(s). Remaining strokes: ${_drawableStrokes.length}';
    });

    if (_animStrokes.isEmpty) {
      _controller.stop();
      setState(() {
        _animValue = 0.0;
        _stepMode = false;
        _stepStrokeCount = 0;
        _activeAnimationDurationSec = 0.0;
        _activeAnimationWorkerCount = 1;
      });
      final completer = _animationCompletionCompleter;
      _animationCompletionCompleter = null;
      if (completer != null && !completer.isCompleted) {
        completer.complete();
      }
    }

    if (_backendEnabled) {
      for (final record in deletedRecords.where((r) => r.syncWithBackend)) {
        () async {
          try {
            await _syncDeleteOnBackend(record.backendDeleteName);
          } catch (e) {
            if (!mounted) return;
            setState(() {
              _status += '\n[Backend delete failed: $e]';
            });
          }
        }();
      }
    }
  }

  Future<void> _animateDeleteObject(String objectId) async {
    final strokes = <DrawableStroke>[
      ..._staticStrokes.where((stroke) => stroke.jsonName == objectId),
      ..._animStrokes.where((stroke) => stroke.jsonName == objectId),
    ];
    if (strokes.isEmpty) {
      return;
    }
    final bounds = _objectBounds(objectId);
    if (bounds == null) {
      return;
    }

    final remaining = strokes.toList(growable: true);
    const steps = 18;
    final radius = math.max(bounds.width, bounds.height) * 0.16;

    for (int step = 0; step < steps && remaining.isNotEmpty; step++) {
      final t = step / math.max(1, steps - 1);
      final sweepY = lerpDouble(bounds.top, bounds.bottom, t) ?? bounds.top;
      final swing = ((step % 2 == 0) ? 0.18 : 0.82) * bounds.width;
      final eraser = Offset(bounds.left + swing, sweepY);

      final removeNow = <DrawableStroke>[];
      for (final stroke in remaining) {
        if ((stroke.centroid - eraser).distance <= radius ||
            stroke.bounds.overlaps(Rect.fromCircle(center: eraser, radius: radius))) {
          removeNow.add(stroke);
        }
      }
      if (removeNow.isEmpty && remaining.isNotEmpty) {
        remaining.sort(
          (a, b) =>
              (a.centroid - eraser).distance.compareTo((b.centroid - eraser).distance),
        );
        removeNow.add(remaining.first);
      }

      remaining.removeWhere(removeNow.contains);
      if (mounted) {
        setState(() {
          _staticStrokes = _staticStrokes.where((stroke) => !removeNow.contains(stroke)).toList(growable: false);
          _animStrokes = _animStrokes.where((stroke) => !removeNow.contains(stroke)).toList(growable: false);
          _drawableStrokes = [..._staticStrokes, ..._animStrokes];
        });
      }
      await Future<void>.delayed(const Duration(milliseconds: 18));
    }
  }

  Future<void> _moveObjectTo({
    required String target,
    required Offset newOrigin,
  }) async {
    final ids = _resolveObjectIds(target);
    if (ids.isEmpty) {
      setState(() {
        _status = 'Move skipped: no object matches "$target".';
      });
      return;
    }

    final startOrigins = <String, Offset>{};
    for (final objectId in ids) {
      final record = _boardObjectsById[objectId];
      if (record != null) {
        startOrigins[objectId] = record.origin;
      }
    }
    if (startOrigins.isEmpty) {
      return;
    }

    for (final objectId in ids) {
      await _animateObjectSaturation(objectId, 0.0, durationMs: 110);
    }

    const steps = 22;
    for (int step = 1; step <= steps; step++) {
      final t = step / steps;
      final eased = _smoothMotion(t);
      final deltas = <String, Offset>{};
      for (final entry in startOrigins.entries) {
        final currentRecord = _boardObjectsById[entry.key];
        if (currentRecord == null) {
          continue;
        }
        final desiredOrigin = Offset.lerp(entry.value, newOrigin, eased) ?? newOrigin;
        deltas[entry.key] = desiredOrigin - currentRecord.origin;
        currentRecord.origin = desiredOrigin;
      }
      if (deltas.isNotEmpty && mounted) {
        setState(() {
          _staticStrokes = _translateObjectsInList(_staticStrokes, deltas);
          _animStrokes = _translateObjectsInList(_animStrokes, deltas);
          _drawableStrokes = [..._staticStrokes, ..._animStrokes];
        });
      }
      await Future<void>.delayed(const Duration(milliseconds: 16));
    }

    for (final objectId in ids) {
      await _animateObjectSaturation(objectId, 1.0, durationMs: 110);
    }

    if (mounted) {
      setState(() {
        _status = 'Moved ${ids.length} object(s) to (${newOrigin.dx}, ${newOrigin.dy}).';
      });
    }
  }

  List<DrawableStroke> _translateObjectsInList(
    List<DrawableStroke> source,
    Map<String, Offset> deltas,
  ) {
    return source.map((stroke) {
      final delta = deltas[stroke.jsonName];
      if (delta == null || delta == Offset.zero) {
        return stroke;
      }
      return _translateStroke(stroke, delta);
    }).toList(growable: false);
  }

  DrawableStroke _translateStroke(DrawableStroke source, Offset delta) {
    if (delta == Offset.zero) {
      return source;
    }

    final movedPoints = source.points
        .map((p) => Offset(p.dx + delta.dx, p.dy + delta.dy))
        .toList(growable: false);
    final movedOriginalPoints = source.originalPoints
        .map((p) => Offset(p.dx + delta.dx, p.dy + delta.dy))
        .toList(growable: false);

    return source.copyWith(
      objectOrigin: Offset(
        source.objectOrigin.dx + delta.dx,
        source.objectOrigin.dy + delta.dy,
      ),
      points: movedPoints,
      originalPoints: movedOriginalPoints,
      centroid: Offset(
        source.centroid.dx + delta.dx,
        source.centroid.dy + delta.dy,
      ),
      bounds: source.bounds.shift(delta),
    );
  }

  Future<void> _linkObjectToImage({
    required String target,
    required String imageName,
  }) async {
    final targetIds = _resolveObjectIds(target);
    final imageIds = _resolveObjectIds(imageName);

    if (targetIds.isEmpty || imageIds.isEmpty) {
      setState(() {
        _status = 'link_to_image skipped: target="$target", image="$imageName"';
      });
      return;
    }

    final linkedTo = imageIds.first;
    for (final targetId in targetIds) {
      final record = _boardObjectsById[targetId];
      if (record != null) {
        record.linkedToObjectId = linkedTo;
      }
    }

    setState(() {
      _status = 'Linked ${targetIds.length} object(s) to $linkedTo.';
    });
  }

  Future<void> _drawCluster({
    required String clusterRef,
    ResolvedDiagramCluster? resolved,
  }) async {
    resolved ??= _resolveCluster(clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'draw_cluster skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    final record = _boardObjectsById[resolved.diagram.objectId];
    if (record == null || record.fileName == null) {
      return;
    }
    final blueprint = await _loadVectorBlueprint(record.fileName!);
    if (blueprint == null) {
      return;
    }

    final missingIndexes = resolved.cluster.strokeIndexes
        .difference(resolved.diagram.drawnStrokeIndexes)
        .toSet();
    if (missingIndexes.isEmpty) {
      return;
    }

    final poly = blueprint.polylines
        .where((stroke) => missingIndexes.contains(stroke.sourceStrokeIndex))
        .toList(growable: false);
    final cubics = blueprint.cubics
        .where((stroke) => missingIndexes.contains(stroke.sourceStrokeIndex))
        .toList(growable: false);

    if (poly.isEmpty && cubics.isEmpty) {
      return;
    }

    final newStrokes = _buildDrawableStrokesForObject(
      jsonName: resolved.diagram.objectId,
      origin: record.origin,
      objectScale: record.scale,
      polylines: poly,
      cubics: cubics,
      srcWidth: record.sourceWidth ?? blueprint.srcWidth,
      srcHeight: record.sourceHeight ?? blueprint.srcHeight,
      targetResolution: _targetResolution,
      basePenWidth: _basePenWidthPx,
    );

    setState(() {
      _finishCurrentAnimIntoStatic();
      _animIsText = false;
      _animStrokes = newStrokes;
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];
      resolved!.cluster.wasExplicitlyDrawn = true;
      resolved.diagram.drawnStrokeIndexes.addAll(missingIndexes);
      _status = 'Drawing cluster ${resolved.cluster.clusterRef}';
    });

    _recomputeTimingForAnimStrokes();
    await _waitForStrokeAnimationComplete();
  }

  Future<void> _highlightCluster({
    required String clusterRef,
    ResolvedDiagramCluster? resolved,
  }) async {
    resolved ??= _resolveCluster(clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'highlight_cluster skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    await _applyHighlightClusterRef(
      diagramObjectId: resolved.diagram.objectId,
      clusterRef: resolved.cluster.clusterRef,
    );
  }

  Future<void> _zoomCluster({
    required String clusterRef,
    ResolvedDiagramCluster? resolved,
  }) async {
    resolved ??= _resolveCluster(clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'zoom_cluster skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    await _activateZoom(
      diagramObjectId: resolved.diagram.objectId,
      clusterRef: resolved.cluster.clusterRef,
      focus: _clusterCenterForResolved(resolved) ??
          (_objectBounds(resolved.diagram.objectId)?.center ?? Offset.zero),
    );
  }

  Future<void> _writeDiagramLabel({
    required String clusterRef,
    required String text,
    ResolvedDiagramCluster? resolved,
    int? actionKey,
  }) async {
    resolved ??= _resolveCluster(clusterRef);
    if (resolved == null) {
      setState(() {
        _status = 'write_label skipped: no cluster for "$clusterRef".';
      });
      return;
    }

    final diagramRecord = _boardObjectsById[resolved.diagram.objectId];
    if (diagramRecord == null) {
      return;
    }

    final writeTarget = await _resolveDiagramWriteTarget(
      diagram: resolved.diagram,
      cluster: resolved.cluster,
      diagramRecord: diagramRecord,
      text: text,
    );

    final labelObjectId = _nextTempObjectId('label');
    await _writeTextPromptLocal(
      prompt: text,
      origin: writeTarget.origin,
      letterSize: writeTarget.letterSize,
      strokeSlowdown: 1.0,
      boardObjectId: labelObjectId,
      logicalName: labelObjectId,
      attachedToObjectId: resolved.diagram.objectId,
      aliases: <String>{labelObjectId, text, clusterRef},
      isTemporary: true,
      ownerDiagramObjectId: resolved.diagram.objectId,
      syncWithBackend: false,
    );
    await _waitForStrokeAnimationComplete();

    final labelBounds = _objectBounds(labelObjectId);
    final clusterBounds = _clusterBoundsForResolved(resolved);
    if (labelBounds != null && clusterBounds != null) {
      final lineSegments = _buildLeaderLineSegments(
        fromBounds: labelBounds,
        toBounds: clusterBounds,
        gap: 12.0,
      );
      await _appendGeneratedPolylinesToObject(
        objectId: labelObjectId,
        polylines: lineSegments,
        color: const Color(0xFF1976D2),
        awaitAnimation: true,
      );
    }

    setState(() {
      resolved!.cluster.lastWrittenLabel = text;
      _diagramLabelsByClusterRef[resolved.cluster.clusterRef] = DiagramPlacedLabel(
        objectId: resolved.diagram.objectId,
        clusterRef: resolved.cluster.clusterRef,
        text: text,
      );
      resolved.diagram.labelWriteCount += 1;
    });
  }

  Future<void> _connectClusters({
    required String fromClusterRef,
    required String toClusterRef,
  }) async {
    // direct calls are routed through the packet aware starter.
    final fromResolved = _resolveCluster(fromClusterRef);
    final toResolved = _resolveCluster(toClusterRef);
    if (fromResolved == null || toResolved == null) {
      return;
    }
    final connectionKey =
        '${fromResolved.cluster.clusterRef}>>${toResolved.cluster.clusterRef}';

    final existing = _diagramConnections.any(
      (connection) => connection.connectionKey == connectionKey,
    );
    if (!existing) {
      setState(() {
        _diagramConnections.add(
          DiagramConnectionRecord(
            connectionKey: connectionKey,
            fromObjectId: fromResolved.diagram.objectId,
            fromClusterRef: fromResolved.cluster.clusterRef,
            toObjectId: toResolved.diagram.objectId,
            toClusterRef: toResolved.cluster.clusterRef,
          ),
        );
      });
    }
  }

  ResolvedDiagramCluster? _resolveCluster(String rawClusterRef) {
    final parsed = DiagramClusterTarget.tryParse(rawClusterRef);
    if (parsed == null) {
      return null;
    }

    final objectIds = _resolveObjectIds(parsed.diagramName);
    for (final objectId in objectIds) {
      final diagram = _diagramStatesByObjectId[objectId];
      if (diagram == null) {
        continue;
      }

      final cluster = diagram.resolveCluster(parsed.clusterName);
      if (cluster != null) {
        return ResolvedDiagramCluster(diagram: diagram, cluster: cluster);
      }
    }

    return null;
  }

  Future<void> _loadDiagramStateIfPresent({
    required String objectId,
    required String diagramName,
    required String processedId,
  }) async {
    final path = '$_clustersFolder\\$processedId.json';
    final file = File(path);
    if (!file.existsSync()) {
      return;
    }

    try {
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map) {
        return;
      }

      final diagramState = DiagramRuntimeState(
        objectId: objectId,
        diagramName: diagramName,
        processedId: processedId,
      );

      final refinedLabelMap = decoded['refined_label_to_strokes'];
      if (refinedLabelMap is Map) {
        for (final entry in refinedLabelMap.entries) {
          final matchedLabel = entry.key.toString().trim();
          if (matchedLabel.isEmpty) {
            continue;
          }

          final labelEntry = entry.value;
          if (labelEntry is! Map) {
            continue;
          }

          final strokeIndexes = _coerceStrokeIndexes(labelEntry['stroke_indexes']);
          if (strokeIndexes.isEmpty) {
            continue;
          }

          final cluster = DiagramClusterRecord(
            clusterRef: '$diagramName : $matchedLabel',
            diagramName: diagramName,
            label: matchedLabel,
            normalizedLabel: _normalizeLookupKey(matchedLabel),
            strokeIndexes: strokeIndexes,
            targetKey:
                (labelEntry['source_key'] ?? labelEntry['target_key'])
                    ?.toString(),
          );

          diagramState.addCluster(cluster);
        }
      } else if (decoded['labels'] is List) {
        final labelsJson = decoded['labels'] as List;
        for (final labelEntry in labelsJson) {
          if (labelEntry is! Map) {
            continue;
          }

          final matchedLabel =
              (labelEntry['matched_label'] ?? '').toString().trim();
          if (matchedLabel.isEmpty) {
            continue;
          }

          final strokeIndexes = _coerceStrokeIndexes(labelEntry['stroke_indexes']);
          if (strokeIndexes.isEmpty) {
            continue;
          }

          final cluster = DiagramClusterRecord(
            clusterRef: '$diagramName : $matchedLabel',
            diagramName: diagramName,
            label: matchedLabel,
            normalizedLabel: _normalizeLookupKey(matchedLabel),
            strokeIndexes: strokeIndexes,
            targetKey: labelEntry['target_key']?.toString(),
          );

          diagramState.addCluster(cluster);
        }
      }

      if (diagramState.clustersByCanonicalRef.isNotEmpty) {
        setState(() {
          _diagramStatesByObjectId[objectId] = diagramState;
        });
      }
    } catch (_) {
      // Ignore malformed diagram label metadata.
    }
  }

  Future<void> _animateDrawCluster({
    required DiagramClusterRecord cluster,
  }) async {}

  Set<int> _coerceStrokeIndexes(dynamic rawValue) {
    if (rawValue is! List) {
      return <int>{};
    }

    final indexes = <int>{};
    for (final entry in rawValue) {
      if (entry is int) {
        indexes.add(entry);
        continue;
      }
      if (entry is num) {
        indexes.add(entry.toInt());
        continue;
      }
      final parsed = int.tryParse(entry.toString());
      if (parsed != null) {
        indexes.add(parsed);
      }
    }
    return indexes;
  }

  Future<void> _animateHighlightCluster({
    required DiagramClusterRecord cluster,
  }) async {}

  Future<void> _animateZoomCluster({
    required DiagramClusterRecord cluster,
  }) async {}

  Future<void> _animateWriteLabel({
    required DiagramClusterRecord cluster,
    required String text,
  }) async {}

  Future<void> _animateConnectClusters({
    required DiagramClusterRecord fromCluster,
    required DiagramClusterRecord toCluster,
  }) async {}

  Future<void> _applyHighlightClusterRef({
    required String diagramObjectId,
    required String clusterRef,
  }) async {
    final refs = _activeHighlightRefsByDiagramId.putIfAbsent(
      diagramObjectId,
      () => <String>{},
    );
    final previousCount = refs.length;
    refs.add(clusterRef);
    _highlightedClusterRefs.add(clusterRef);
    _highlightRefCountsByClusterRef[clusterRef] =
        (_highlightRefCountsByClusterRef[clusterRef] ?? 0) + 1;
    if (previousCount == 0) {
      await _animateDiagramNonFocusSaturation(
        diagramObjectId: diagramObjectId,
        target: 0.5,
      );
    } else {
      setState(() {});
    }
  }

  Future<void> _removeHighlightClusterRef({
    required String diagramObjectId,
    required String clusterRef,
  }) async {
    final currentCount = _highlightRefCountsByClusterRef[clusterRef] ?? 0;
    if (currentCount <= 1) {
      _highlightRefCountsByClusterRef.remove(clusterRef);
      _highlightedClusterRefs.remove(clusterRef);
    } else {
      _highlightRefCountsByClusterRef[clusterRef] = currentCount - 1;
    }

    final refs = _activeHighlightRefsByDiagramId[diagramObjectId];
    if (refs != null) {
      final stillAnyForRef = (_highlightRefCountsByClusterRef[clusterRef] ?? 0) > 0;
      if (!stillAnyForRef) {
        refs.remove(clusterRef);
      }
      if (refs.isEmpty) {
        _activeHighlightRefsByDiagramId.remove(diagramObjectId);
        await _animateDiagramNonFocusSaturation(
          diagramObjectId: diagramObjectId,
          target: 1.0,
        );
        return;
      }
    }
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _activateZoom({
    required String diagramObjectId,
    required String clusterRef,
    required Offset focus,
  }) async {
    _zoomedDiagramObjectId = diagramObjectId;
    _zoomedClusterRef = clusterRef;
    final startScale = _boardZoomScale;
    final startFocus = _boardZoomWorldCenter;
    final startSat = _diagramNonFocusSaturationByObjectId[diagramObjectId] ?? 1.0;
    const targetScale = 1.5;
    const targetSat = 0.0;
    const steps = 20;
    for (int i = 1; i <= steps; i++) {
      final t = _smoothMotion(i / steps);
      if (!mounted) break;
      setState(() {
        _boardZoomScale = lerpDouble(startScale, targetScale, t) ?? targetScale;
        _boardZoomWorldCenter = Offset.lerp(startFocus, focus, t) ?? focus;
        _diagramNonFocusSaturationByObjectId[diagramObjectId] =
            lerpDouble(startSat, targetSat, t) ?? targetSat;
      });
      await Future<void>.delayed(const Duration(milliseconds: 16));
    }
  }

  Future<void> _deactivateZoom({
    required String diagramObjectId,
    required String clusterRef,
  }) async {
    if (_zoomedDiagramObjectId != diagramObjectId || _zoomedClusterRef != clusterRef) {
      return;
    }
    final startScale = _boardZoomScale;
    final startFocus = _boardZoomWorldCenter;
    final startSat = _diagramNonFocusSaturationByObjectId[diagramObjectId] ?? 0.0;
    const steps = 18;
    for (int i = 1; i <= steps; i++) {
      final t = _smoothMotion(i / steps);
      if (!mounted) break;
      setState(() {
        _boardZoomScale = lerpDouble(startScale, 1.0, t) ?? 1.0;
        _boardZoomWorldCenter = Offset.lerp(startFocus, Offset.zero, t) ?? Offset.zero;
        _diagramNonFocusSaturationByObjectId[diagramObjectId] =
            lerpDouble(startSat, 1.0, t) ?? 1.0;
      });
      await Future<void>.delayed(const Duration(milliseconds: 16));
    }
    _zoomedDiagramObjectId = null;
    _zoomedClusterRef = null;
    if (mounted) {
      setState(() {});
    }
  }

  Future<void> _animateDiagramNonFocusSaturation({
    required String diagramObjectId,
    required double target,
  }) async {
    final start = _diagramNonFocusSaturationByObjectId[diagramObjectId] ?? 1.0;
    const steps = 14;
    for (int i = 1; i <= steps; i++) {
      final t = _smoothMotion(i / steps);
      if (!mounted) break;
      setState(() {
        _diagramNonFocusSaturationByObjectId[diagramObjectId] =
            lerpDouble(start, target, t) ?? target;
      });
      await Future<void>.delayed(const Duration(milliseconds: 16));
    }
  }

  Future<void> _animateObjectSaturation(
    String objectId,
    double target, {
    int durationMs = 120,
  }) async {
    final start = _objectSaturationById[objectId] ?? 1.0;
    final steps = math.max(4, durationMs ~/ 16);
    for (int i = 1; i <= steps; i++) {
      final t = _smoothMotion(i / steps);
      if (!mounted) break;
      setState(() {
        _objectSaturationById[objectId] = lerpDouble(start, target, t) ?? target;
      });
      await Future<void>.delayed(const Duration(milliseconds: 16));
    }
  }

  double _smoothMotion(double t) {
    t = t.clamp(0.0, 1.0).toDouble();
    return t * t * (3.0 - 2.0 * t);
  }

  Future<void> _waitForStrokeAnimationComplete() async {
    final completer = _animationCompletionCompleter;
    if (completer != null) {
      await completer.future;
    }
  }

  void _finishCurrentAnimIntoStatic() {
    if (_animStrokes.isNotEmpty) {
      _controller.stop();
      _staticStrokes = [..._staticStrokes, ..._animStrokes];
      _animStrokes = const [];
      _animValue = 0.0;
      _drawableStrokes = [..._staticStrokes];
      final completer = _animationCompletionCompleter;
      _animationCompletionCompleter = null;
      if (completer != null && !completer.isCompleted) {
        completer.complete();
      }
    }
  }

  Future<void> _addGeneratedPolylineObject({
    required String objectId,
    required String displayName,
    required List<List<Offset>> polylines,
    required Color color,
    required bool isTemporary,
    required bool syncWithBackend,
    String? ownerDiagramObjectId,
    Set<String>? aliases,
    bool awaitAnimation = false,
  }) async {
    await _deleteObject(id: objectId, silentIfMissing: true, animate: false);
    final bounds = _boundsOfPolylineGroups(polylines);
    final origin = bounds.center;
    final diagBoard =
        math.sqrt(_boardWidth * _boardWidth + _boardHeight * _boardHeight);
    final strokes = <DrawableStroke>[];
    for (int i = 0; i < polylines.length; i++) {
      final points = polylines[i];
      if (points.length < 2) {
        continue;
      }
      strokes.add(
        _makeDrawableFromPoints(
          jsonName: objectId,
          objectOrigin: origin,
          objectScale: 1.0,
          pts: points,
          basePenWidth: _basePenWidthPx,
          diag: diagBoard,
          sourceStrokeIndex: i,
          strokeColor: color,
        ),
      );
    }
    if (strokes.isEmpty) {
      return;
    }

    final record = BoardObjectRecord(
      objectId: objectId,
      kind: BoardObjectKind.image,
      displayName: displayName,
      origin: origin,
      scale: 1.0,
      fileName: null,
      processedId: null,
      aliases: <String>{displayName, objectId, ...?aliases},
      ownerDiagramObjectId: ownerDiagramObjectId,
      isTemporary: isTemporary,
      syncWithBackend: syncWithBackend,
      sourceBounds: bounds,
      sourceWidth: bounds.width,
      sourceHeight: bounds.height,
    );

    if (!mounted) return;
    setState(() {
      _finishCurrentAnimIntoStatic();
      _animIsText = false;
      _animStrokes = strokes;
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];
      _registerObject(record);
    });
    _recomputeTimingForAnimStrokes();
    if (awaitAnimation) {
      await _waitForStrokeAnimationComplete();
    }
  }

  Future<void> _appendGeneratedPolylinesToObject({
    required String objectId,
    required List<List<Offset>> polylines,
    required Color color,
    bool awaitAnimation = false,
  }) async {
    final record = _boardObjectsById[objectId];
    if (record == null) {
      return;
    }
    final diagBoard =
        math.sqrt(_boardWidth * _boardWidth + _boardHeight * _boardHeight);
    final baseIndex = <DrawableStroke>[
      ..._staticStrokes.where((stroke) => stroke.jsonName == objectId),
      ..._animStrokes.where((stroke) => stroke.jsonName == objectId),
    ].length;
    final strokes = <DrawableStroke>[];
    for (int i = 0; i < polylines.length; i++) {
      final points = polylines[i];
      if (points.length < 2) {
        continue;
      }
      strokes.add(
        _makeDrawableFromPoints(
          jsonName: objectId,
          objectOrigin: record.origin,
          objectScale: record.scale,
          pts: points,
          basePenWidth: _basePenWidthPx,
          diag: diagBoard,
          sourceStrokeIndex: baseIndex + i,
          strokeColor: color,
        ),
      );
    }
    if (strokes.isEmpty || !mounted) {
      return;
    }
    setState(() {
      _finishCurrentAnimIntoStatic();
      _animIsText = false;
      _animStrokes = strokes;
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];
    });
    _recomputeTimingForAnimStrokes();
    if (awaitAnimation) {
      await _waitForStrokeAnimationComplete();
    }
  }

  Rect _boundsOfPolylineGroups(List<List<Offset>> polylines) {
    final pts = polylines.expand((group) => group).toList(growable: false);
    return _boundsOfPoints(pts);
  }

  List<List<Offset>> _buildDoubleArrowConnectorSegments({
    required Rect fromBounds,
    required Rect toBounds,
    required double gap,
    required double arrowLength,
    required double arrowHalfWidth,
  }) {
    final fromCenter = fromBounds.center;
    final toCenter = toBounds.center;
    final trimmed = _trimLineBetweenBounds(
      fromBounds: fromBounds,
      toBounds: toBounds,
      extraGap: gap,
    );
    final start = trimmed.$1;
    final end = trimmed.$2;
    final dir = (end - start);
    final len = dir.distance;
    if (len <= 1e-6) {
      return const <List<Offset>>[];
    }
    final unit = Offset(dir.dx / len, dir.dy / len);
    final normal = Offset(-unit.dy, unit.dx);

    final startArrowBase = start + unit * arrowLength;
    final endArrowBase = end - unit * arrowLength;

    return <List<Offset>>[
      <Offset>[start, end],
      <Offset>[start, startArrowBase + normal * arrowHalfWidth],
      <Offset>[start, startArrowBase - normal * arrowHalfWidth],
      <Offset>[end, endArrowBase + normal * arrowHalfWidth],
      <Offset>[end, endArrowBase - normal * arrowHalfWidth],
    ];
  }

  List<List<Offset>> _buildLeaderLineSegments({
    required Rect fromBounds,
    required Rect toBounds,
    required double gap,
  }) {
    final trimmed = _trimLineBetweenBounds(
      fromBounds: fromBounds,
      toBounds: toBounds,
      extraGap: gap,
    );
    return <List<Offset>>[
      <Offset>[trimmed.$1, trimmed.$2],
    ];
  }

  List<List<Offset>> _buildDottedLineSegments({
    required Offset from,
    required Offset to,
    required double gapFromStart,
    required double gapFromEnd,
    required double dashLength,
    required double dashGap,
  }) {
    final vector = to - from;
    final len = vector.distance;
    if (len <= gapFromStart + gapFromEnd + dashLength) {
      return <List<Offset>>[];
    }
    final unit = Offset(vector.dx / len, vector.dy / len);
    final start = from + unit * gapFromStart;
    final end = to - unit * gapFromEnd;
    final usable = (end - start).distance;
    final segments = <List<Offset>>[];
    double cursor = 0.0;
    while (cursor < usable) {
      final dashEnd = math.min(cursor + dashLength, usable);
      final p0 = start + unit * cursor;
      final p1 = start + unit * dashEnd;
      if ((p1 - p0).distance > 1e-3) {
        segments.add(<Offset>[p0, p1]);
      }
      cursor = dashEnd + dashGap;
    }
    return segments;
  }

  (Offset, Offset) _trimLineBetweenBounds({
    required Rect fromBounds,
    required Rect toBounds,
    required double extraGap,
  }) {
    final fromCenter = fromBounds.center;
    final toCenter = toBounds.center;
    final dir = toCenter - fromCenter;
    final len = dir.distance;
    if (len <= 1e-6) {
      return (fromCenter, toCenter);
    }
    final unit = Offset(dir.dx / len, dir.dy / len);
    final start = _lineExitPoint(fromBounds, fromCenter, unit, extraGap);
    final end = _lineExitPoint(toBounds, toCenter, Offset(-unit.dx, -unit.dy), extraGap);
    return (start, end);
  }

  Offset _lineExitPoint(Rect bounds, Offset center, Offset direction, double extraGap) {
    final dir = direction;
    final candidates = <double>[];
    if (dir.dx.abs() > 1e-6) {
      candidates.add((bounds.left - center.dx) / dir.dx);
      candidates.add((bounds.right - center.dx) / dir.dx);
    }
    if (dir.dy.abs() > 1e-6) {
      candidates.add((bounds.top - center.dy) / dir.dy);
      candidates.add((bounds.bottom - center.dy) / dir.dy);
    }
    double bestT = 0.0;
    for (final t in candidates) {
      if (t <= 0) continue;
      final point = center + dir * t;
      if (point.dx >= bounds.left - 1e-3 &&
          point.dx <= bounds.right + 1e-3 &&
          point.dy >= bounds.top - 1e-3 &&
          point.dy <= bounds.bottom + 1e-3) {
        if (bestT == 0.0 || t < bestT) {
          bestT = t;
        }
      }
    }
    final edgePoint = center + dir * bestT;
    return edgePoint + dir * extraGap;
  }

  String _nextTempObjectId(String prefix) {
    _tempObjectCounter += 1;
    return '${prefix}_temp_${_tempObjectCounter}';
  }

  Future<VectorBlueprint?> _loadVectorBlueprint(String fileName) async {
    final cached = _vectorBlueprintCache[fileName];
    if (cached != null) {
      return cached;
    }

    final path = '$_vectorsFolder\\$fileName';
    final file = File(path);
    if (!file.existsSync()) {
      return null;
    }
    try {
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map || decoded['strokes'] is! List) {
        return null;
      }
      final format =
          (decoded['vector_format'] as String?)?.toLowerCase() ?? 'polyline';
      final strokesJson = decoded['strokes'] as List;
      final poly = <StrokePolyline>[];
      final cubics = <StrokeCubic>[];
      final srcWidth = (decoded['width'] as num?)?.toDouble() ?? 1000.0;
      final srcHeight = (decoded['height'] as num?)?.toDouble() ?? 1000.0;

      if (format == 'bezier_cubic') {
        for (int i = 0; i < strokesJson.length; i++) {
          final s = strokesJson[i];
          if (s is! Map || s['segments'] is! List) continue;
          final color = _parseStrokeColor(s);
          final segs = <CubicSegment>[];
          for (final seg in s['segments'] as List) {
            if (seg is List && seg.length >= 8) {
              segs.add(
                CubicSegment(
                  p0: Offset((seg[0] as num).toDouble(), (seg[1] as num).toDouble()),
                  c1: Offset((seg[2] as num).toDouble(), (seg[3] as num).toDouble()),
                  c2: Offset((seg[4] as num).toDouble(), (seg[5] as num).toDouble()),
                  p1: Offset((seg[6] as num).toDouble(), (seg[7] as num).toDouble()),
                ),
              );
            }
          }
          if (segs.isNotEmpty) {
            cubics.add(StrokeCubic(segs, color: color, sourceStrokeIndex: i));
          }
        }
      } else {
        for (int i = 0; i < strokesJson.length; i++) {
          final s = strokesJson[i];
          if (s is! Map || s['points'] is! List) continue;
          final color = _parseStrokeColor(s);
          final points = <Offset>[];
          for (final p in s['points'] as List) {
            if (p is List && p.length >= 2) {
              points.add(Offset((p[0] as num).toDouble(), (p[1] as num).toDouble()));
            }
          }
          if (points.length >= 2) {
            poly.add(StrokePolyline(points, color: color, sourceStrokeIndex: i));
          }
        }
      }

      final sourceBounds = _computeRawBounds(poly, cubics);
      final blueprint = VectorBlueprint(
        fileName: fileName,
        polylines: poly,
        cubics: cubics,
        srcWidth: srcWidth,
        srcHeight: srcHeight,
        sourceBounds: sourceBounds,
      );
      _vectorBlueprintCache[fileName] = blueprint;
      return blueprint;
    } catch (_) {
      return null;
    }
  }

  Future<List<DiagramLabelAnchor>> _loadDiagramLabelAnchors(String processedId) async {
    final cached = _labelAnchorCacheByProcessedId[processedId];
    if (cached != null) {
      return cached;
    }
    final path = '$_processedImagesFolder\\$processedId.json';
    final file = File(path);
    if (!file.existsSync()) {
      _labelAnchorCacheByProcessedId[processedId] = const <DiagramLabelAnchor>[];
      return const <DiagramLabelAnchor>[];
    }
    try {
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map || decoded['words'] is! List) {
        _labelAnchorCacheByProcessedId[processedId] = const <DiagramLabelAnchor>[];
        return const <DiagramLabelAnchor>[];
      }
      final anchors = <DiagramLabelAnchor>[];
      for (final entry in decoded['words'] as List) {
        if (entry is! Map) continue;
        final text = (entry['text'] ?? '').toString().trim();
        if (text.isEmpty) continue;
        Rect? anchorRect;
        final bboxAnchor = entry['bbox_anchor'];
        final bboxMask = entry['bbox_mask'];
        if (bboxAnchor is List && bboxAnchor.length >= 4) {
          anchorRect = Rect.fromLTRB(
            (bboxAnchor[0] as num).toDouble(),
            (bboxAnchor[1] as num).toDouble(),
            (bboxAnchor[2] as num).toDouble(),
            (bboxAnchor[3] as num).toDouble(),
          );
        } else if (bboxMask is List && bboxMask.length >= 4) {
          anchorRect = Rect.fromLTRB(
            (bboxMask[0] as num).toDouble(),
            (bboxMask[1] as num).toDouble(),
            (bboxMask[2] as num).toDouble(),
            (bboxMask[3] as num).toDouble(),
          );
        }
        if (anchorRect != null) {
          anchors.add(DiagramLabelAnchor(
            rawText: text,
            normalizedText: _normalizeLookupKey(text),
            sourceRect: anchorRect,
          ));
        }
      }
      _labelAnchorCacheByProcessedId[processedId] = anchors;
      return anchors;
    } catch (_) {
      _labelAnchorCacheByProcessedId[processedId] = const <DiagramLabelAnchor>[];
      return const <DiagramLabelAnchor>[];
    }
  }

  Future<DiagramWriteTarget> _resolveDiagramWriteTarget({
    required DiagramRuntimeState diagram,
    required DiagramClusterRecord cluster,
    required BoardObjectRecord diagramRecord,
    required String text,
  }) async {
    final anchors = await _loadDiagramLabelAnchors(diagram.processedId);
    final clusterNormalized = _normalizeLookupKey(cluster.label);
    DiagramLabelAnchor? matched;
    for (final anchor in anchors) {
      if (anchor.normalizedText == clusterNormalized ||
          anchor.normalizedText.contains(clusterNormalized) ||
          clusterNormalized.contains(anchor.normalizedText)) {
        matched = anchor;
        break;
      }
    }

    if (matched != null && diagramRecord.sourceBounds != null) {
      final worldRect = _mapSourceRectToWorld(diagramRecord, matched.sourceRect);
      return DiagramWriteTarget(
        origin: Offset(worldRect.left, worldRect.center.dy),
        letterSize: math.max(26.0, worldRect.height * 0.8),
        preferredRect: worldRect,
      );
    }

    final diagramBounds = _objectBounds(diagram.objectId) ??
        Rect.fromCenter(center: diagramRecord.origin, width: 300, height: 220);
    final yOffset = 40.0 + diagram.labelWriteCount * 78.0;
    final fallbackRect = Rect.fromLTWH(
      diagramBounds.right + 40.0,
      diagramBounds.top + yOffset,
      math.max(160.0, diagramBounds.width * 0.25),
      62.0,
    );
    return DiagramWriteTarget(
      origin: Offset(fallbackRect.left, fallbackRect.center.dy),
      letterSize: fallbackRect.height * 0.75,
      preferredRect: fallbackRect,
    );
  }

  Rect _mapSourceRectToWorld(BoardObjectRecord record, Rect sourceRect) {
    final worldTopLeft = _mapSourcePointToWorld(record, sourceRect.topLeft);
    final worldBottomRight = _mapSourcePointToWorld(record, sourceRect.bottomRight);
    return Rect.fromPoints(worldTopLeft, worldBottomRight);
  }

  Offset _mapSourcePointToWorld(BoardObjectRecord record, Offset sourcePoint) {
    final sourceWidth = record.sourceWidth ?? 1000.0;
    final sourceHeight = record.sourceHeight ?? 1000.0;
    final sourceBounds = record.sourceBounds ?? Rect.fromLTWH(0, 0, sourceWidth, sourceHeight);
    final srcMax = math.max(sourceWidth, sourceHeight);
    final baseUpscale = srcMax > 0 ? _targetResolution / srcMax : 1.0;
    final scale = record.scale <= 0 ? 1.0 : record.scale;
    final upscale = baseUpscale * scale;
    final centerScaled = Offset(
      sourceBounds.center.dx * upscale,
      sourceBounds.center.dy * upscale,
    );
    final scaled = Offset(sourcePoint.dx * upscale, sourcePoint.dy * upscale);
    return Offset(
      scaled.dx - centerScaled.dx + record.origin.dx,
      scaled.dy - centerScaled.dy + record.origin.dy,
    );
  }
  Future<void> _ensureFontMetricsLoaded() async {
    if (_fontLineHeightPx != null && _fontImageHeightPx != null) return;
    try {
      final file = File(_fontMetricsPath);
      if (!file.existsSync()) {
        _fontImageHeightPx = _targetResolution;
        _fontLineHeightPx = _targetResolution * 0.5;
        return;
      }
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is Map) {
        final lh = (decoded['line_height_px'] as num?)?.toDouble();
        final ih = (decoded['image_height'] as num?)?.toDouble();
        if (lh != null && lh > 0) _fontLineHeightPx = lh;
        if (ih != null && ih > 0) _fontImageHeightPx = ih;
      }
      _fontLineHeightPx ??= _targetResolution * 0.5;
      _fontImageHeightPx ??= _targetResolution;
    } catch (_) {
      _fontLineHeightPx = _targetResolution * 0.5;
      _fontImageHeightPx = _targetResolution;
    }
  }

  Future<void> _syncCreateImageOnBackend({
    required String fileName,
    required Offset origin,
    required double scale,
  }) async {
    if (!_backendEnabled) return;
    final uri = _apiUri('/api/whiteboard/objects/image/');
    final body = json.encode({
      'file_name': fileName,
      'x': origin.dx,
      'y': origin.dy,
      'scale': scale,
    });

    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: body,
    );
    if (resp.statusCode >= 400) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  Future<void> _syncCreateTextOnBackend({
    required String prompt,
    required Offset origin,
    required double letterSize,
    required double letterGap,
  }) async {
    if (!_backendEnabled) return;
    final uri = _apiUri('/api/whiteboard/objects/text/');
    final body = json.encode({
      'prompt': prompt,
      'x': origin.dx,
      'y': origin.dy,
      'letter_size': letterSize,
      'letter_gap': letterGap,
    });

    final resp = await http.post(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: body,
    );
    if (resp.statusCode >= 400) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  Future<void> _syncDeleteOnBackend(String name) async {
    if (!_backendEnabled) return;
    final uri = _apiUri('/api/whiteboard/objects/delete/');
    final body = json.encode({'name': name});

    final resp = await http.delete(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: body,
    );

    if (resp.statusCode >= 400 && resp.statusCode != 404) {
      throw Exception('HTTP ${resp.statusCode}: ${resp.body}');
    }
  }

  Future<void> _loadObjectsFromBackend() async {
    if (!_backendEnabled) return;
    try {
      final uri = _apiUri('/api/whiteboard/objects/');
      final resp = await http.get(uri);
      if (resp.statusCode != 200) {
        setState(() {
          _status = 'Backend list error: HTTP ${resp.statusCode}';
        });
        return;
      }

      final decoded = json.decode(resp.body);
      if (decoded is! Map || decoded['objects'] is! List) {
        setState(() {
          _status = 'Backend list invalid format';
        });
        return;
      }

      final objs = decoded['objects'] as List;

      for (int index = 0; index < objs.length; index++) {
        final o = objs[index];
        if (o is! Map) continue;
        final name = (o['name'] ?? '').toString();
        final kind = (o['kind'] ?? '').toString();
        final x = (o['pos_x'] as num?)?.toDouble() ?? 0.0;
        final y = (o['pos_y'] as num?)?.toDouble() ?? 0.0;
        final scale =
            (o['scale'] as num?)?.toDouble() ?? _defaultImageObjectScale;

        if (kind == 'image') {
          await _addObjectFromJsonInternal(
            fileName: name,
            origin: Offset(x, y),
            objectScale: scale,
            boardObjectId: _stemWithoutJson(name),
            logicalName: _stemWithoutJson(name),
            aliases: <String>{name, _stemWithoutJson(name)},
          );
        } else if (kind == 'text') {
          final letterSize =
              (o['letter_size'] as num?)?.toDouble() ?? _textBaseFontSizeRef;
          final letterGap =
              (o['letter_gap'] as num?)?.toDouble() ?? _textLetterGapPx;
          _textLetterGapPx = letterGap;
          await _writeTextPromptLocal(
            prompt: name,
            origin: Offset(x, y),
            letterSize: letterSize,
            strokeSlowdown: 1.0,
            boardObjectId: 'backend_text_$index',
            logicalName: name,
            aliases: <String>{name},
          );
        }
      }

      if (mounted) {
        _controller.stop();
        setState(() {
          _staticStrokes = [..._staticStrokes, ..._animStrokes];
          _animStrokes = const [];
          _drawableStrokes = [..._staticStrokes];
          _animValue = 1.0;
          _status = 'Synced ${objs.length} object(s) from backend.';
        });
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _status = 'Backend sync failed: $e';
      });
    }
  }

  Color _parseStrokeColor(dynamic strokeJson) {
    if (strokeJson is! Map) return Colors.black;

    final rgb = strokeJson['color_rgb'] ??
        strokeJson['colour_rgb'] ??
        strokeJson['rgb'] ??
        strokeJson['color'] ??
        strokeJson['stroke_color'];

    if (rgb is List && rgb.length >= 3) {
      double r = (rgb[0] as num).toDouble();
      double g = (rgb[1] as num).toDouble();
      double b = (rgb[2] as num).toDouble();

      if (r <= 1.0 && g <= 1.0 && b <= 1.0) {
        r *= 255.0;
        g *= 255.0;
        b *= 255.0;
      }

      final int ri = r.round().clamp(0, 255).toInt();
      final int gi = g.round().clamp(0, 255).toInt();
      final int bi = b.round().clamp(0, 255).toInt();
      return Color.fromARGB(255, ri, gi, bi);
    }

    final hex =
        strokeJson['color_hex'] ?? strokeJson['colour_hex'] ?? strokeJson['hex'];
    if (hex is String && hex.isNotEmpty) {
      var s = hex.trim();
      if (s.startsWith('#')) s = s.substring(1);
      if (s.length == 6) {
        final v = int.tryParse(s, radix: 16);
        if (v != null) return Color(0xFF000000 | v);
      }
    }

    final ci = strokeJson['color_int'] ?? strokeJson['colour_int'];
    if (ci is int) {
      if ((ci & 0xFF000000) == 0) return Color(0xFF000000 | ci);
      return Color(ci);
    }

    return Colors.black;
  }

  Future<void> _loadAndRender() async {
    final fileName = _fileNameController.text.trim();
    if (fileName.isEmpty) {
      setState(() {
        _status = 'No file name specified.';
      });
      return;
    }

    final x = double.tryParse(_posXController.text.trim()) ?? 0.0;
    final y = double.tryParse(_posYController.text.trim()) ?? 0.0;
    final scale =
        double.tryParse(_scaleController.text.trim()) ??
        _defaultImageObjectScale;

    await _whiteboardActions.drawImage(
      fileName: fileName,
      origin: Offset(x, y),
      objectScale: scale,
      boardObjectId: _stemWithoutJson(fileName),
      logicalName: _stemWithoutJson(fileName),
      aliases: <String>{fileName, _stemWithoutJson(fileName)},
    );
  }

  Future<void> _addObjectFromJson({
    required String fileName,
    required Offset origin,
    required double objectScale,
    String? boardObjectId,
    String? logicalName,
    String? processedId,
    Set<String>? aliases,
  }) async {
    await _addObjectFromJsonInternal(
      fileName: fileName,
      origin: origin,
      objectScale: objectScale,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      processedId: processedId,
      aliases: aliases,
    );

    if (_backendEnabled) {
      () async {
        try {
          await _syncCreateImageOnBackend(
            fileName: fileName,
            origin: origin,
            scale: objectScale,
          );
        } catch (e) {
          if (!mounted) return;
          setState(() {
            _status += '\n[Backend create image failed: $e]';
          });
        }
      }();
    }
  }

  Future<void> _addObjectFromJsonInternal({
    required String fileName,
    required Offset origin,
    required double objectScale,
    String? boardObjectId,
    String? logicalName,
    String? processedId,
    Set<String>? aliases,
  }) async {
    final path = '$_vectorsFolder\\$fileName';
    setState(() => _status = 'Loading $fileName…');

    final file = File(path);
    if (!file.existsSync()) {
      setState(() => _status = 'Not found:\n$path');
      return;
    }

    try {
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map || decoded['strokes'] is! List) {
        setState(() => _status = 'Invalid JSON format (no "strokes").');
        return;
      }

      final format =
          (decoded['vector_format'] as String?)?.toLowerCase() ?? 'polyline';
      final strokesJson = decoded['strokes'] as List;

      final poly = <StrokePolyline>[];
      final cubics = <StrokeCubic>[];

      final srcWidth = (decoded['width'] as num?)?.toDouble();
      final srcHeight = (decoded['height'] as num?)?.toDouble();
      _srcWidth = srcWidth;
      _srcHeight = srcHeight;

      if (format == 'bezier_cubic') {
        for (int i = 0; i < strokesJson.length; i++) {
          final s = strokesJson[i];
          if (s is! Map || s['segments'] is! List) continue;

          final color = _parseStrokeColor(s);
          final segsJson = s['segments'] as List;
          final segs = <CubicSegment>[];
          for (final seg in segsJson) {
            if (seg is List && seg.length >= 8) {
              final p0 = Offset(
                (seg[0] as num).toDouble(),
                (seg[1] as num).toDouble(),
              );
              final c1 = Offset(
                (seg[2] as num).toDouble(),
                (seg[3] as num).toDouble(),
              );
              final c2 = Offset(
                (seg[4] as num).toDouble(),
                (seg[5] as num).toDouble(),
              );
              final p1 = Offset(
                (seg[6] as num).toDouble(),
                (seg[7] as num).toDouble(),
              );
              segs.add(CubicSegment(p0: p0, c1: c1, c2: c2, p1: p1));
            }
          }
          if (segs.isNotEmpty) {
            cubics.add(StrokeCubic(
              segs,
              color: color,
              sourceStrokeIndex: i,
            ));
          }
        }
      } else {
        for (int i = 0; i < strokesJson.length; i++) {
          final s = strokesJson[i];
          if (s is! Map || s['points'] is! List) continue;

          final color = _parseStrokeColor(s);
          final pts = s['points'] as List;
          final points = <Offset>[];
          for (final p in pts) {
            if (p is List && p.length >= 2) {
              points.add(
                Offset((p[0] as num).toDouble(), (p[1] as num).toDouble()),
              );
            }
          }
          if (points.length >= 2) {
            poly.add(StrokePolyline(
              points,
              color: color,
              sourceStrokeIndex: i,
            ));
          }
        }
      }

      _polyStrokes = poly;
      _cubicStrokes = cubics;

      double useWidth = srcWidth ?? 1000.0;
      double useHeight = srcHeight ?? 1000.0;
      if ((srcWidth == null || srcHeight == null) &&
          (poly.isNotEmpty || cubics.isNotEmpty)) {
        final bounds = _computeRawBounds(poly, cubics);
        useWidth = bounds.width;
        useHeight = bounds.height;
        _srcWidth = useWidth;
        _srcHeight = useHeight;
      }

      final objectId = (boardObjectId?.trim().isNotEmpty == true)
          ? boardObjectId!.trim()
          : (processedId?.trim().isNotEmpty == true)
              ? processedId!.trim()
              : _stemWithoutJson(fileName);

      final primaryName = (logicalName?.trim().isNotEmpty == true)
          ? logicalName!.trim()
          : _stemWithoutJson(fileName);

      final aliasSet = <String>{
        primaryName,
        fileName,
        _stemWithoutJson(fileName),
        if ((processedId ?? '').trim().isNotEmpty) processedId!.trim(),
        ...?aliases,
      };

      await _deleteObject(id: objectId, silentIfMissing: true, animate: false);

      final newStrokes = _buildDrawableStrokesForObject(
        jsonName: objectId,
        origin: origin,
        objectScale: objectScale,
        polylines: poly,
        cubics: cubics,
        srcWidth: useWidth,
        srcHeight: useHeight,
        targetResolution: _targetResolution,
        basePenWidth: _basePenWidthPx,
      );

      final record = BoardObjectRecord(
        objectId: objectId,
        kind: BoardObjectKind.image,
        displayName: primaryName,
        origin: origin,
        scale: objectScale,
        fileName: fileName,
        processedId: processedId,
        aliases: aliasSet,
        sourceWidth: useWidth,
        sourceHeight: useHeight,
        sourceBounds: _computeRawBounds(poly, cubics),
      );

      setState(() {
        if (_animStrokes.isNotEmpty) {
          _controller.stop();
          _staticStrokes = [..._staticStrokes, ..._animStrokes];
          _animStrokes = const [];
          _animValue = 0.0;
        }

        _animIsText = false;
        _animStrokes = newStrokes;
        _drawableStrokes = [..._staticStrokes, ..._animStrokes];
        _registerObject(record);

        final polyPts = poly.fold<int>(0, (s, e) => s + e.points.length);
        final cubicSegs = cubics.fold<int>(0, (s, e) => s + e.segments.length);
        _status =
            'Added $primaryName\nFormat: $format | strokes: poly=${poly.length}, cubic=${cubics.length}, pts=$polyPts, segs=$cubicSegs\nTotal drawable strokes: ${_drawableStrokes.length}';
      });

      if ((processedId ?? '').trim().isNotEmpty) {
        await _loadDiagramStateIfPresent(
          objectId: objectId,
          diagramName: primaryName,
          processedId: processedId!.trim(),
        );
      }

      _recomputeTimingForAnimStrokes();
    } catch (e, st) {
      setState(() => _status = 'Error loading $fileName: $e');
      print(st);
    }
  }

  static String _resolveBackendSubdir(String subdir) {
    var dir = Directory.current;

    for (int i = 0; i < 10; i++) {
      final candidate = Directory('${dir.path}\\whiteboard_backend');
      if (candidate.existsSync()) {
        if (subdir.isEmpty) return candidate.path;
        return '${candidate.path}\\$subdir';
      }

      final parent = dir.parent;
      if (parent.path == dir.path) {
        break;
      }
      dir = parent;
    }

    return '${Directory.current.path}\\whiteboard_backend\\$subdir';
  }

  Future<GlyphData?> _getGlyphForCode(int codeUnit) async {
    if (_glyphCache.containsKey(codeUnit)) {
      return _glyphCache[codeUnit];
    }

    final hex = codeUnit.toRadixString(16).padLeft(4, '0');
    final path = '$_fontVectorsFolder\\$hex.json';
    final file = File(path);
    if (!file.existsSync()) {
      return null;
    }

    try {
      final raw = await file.readAsString();
      final decoded = json.decode(raw);
      if (decoded is! Map || decoded['strokes'] is! List) {
        return null;
      }

      final strokesJson = decoded['strokes'] as List;
      final format =
          (decoded['vector_format'] as String?)?.toLowerCase() ?? 'bezier_cubic';

      final cubics = <StrokeCubic>[];

      if (format == 'bezier_cubic') {
        for (final s in strokesJson) {
          if (s is! Map || s['segments'] is! List) continue;
          final segsJson = s['segments'] as List;
          final segs = <CubicSegment>[];
          for (final seg in segsJson) {
            if (seg is List && seg.length >= 8) {
              final p0 = Offset(
                (seg[0] as num).toDouble(),
                (seg[1] as num).toDouble(),
              );
              final c1 = Offset(
                (seg[2] as num).toDouble(),
                (seg[3] as num).toDouble(),
              );
              final c2 = Offset(
                (seg[4] as num).toDouble(),
                (seg[5] as num).toDouble(),
              );
              final p1 = Offset(
                (seg[6] as num).toDouble(),
                (seg[7] as num).toDouble(),
              );
              segs.add(CubicSegment(p0: p0, c1: c1, c2: c2, p1: p1));
            }
          }
          if (segs.isNotEmpty) {
            cubics.add(StrokeCubic(
              segs,
              sourceStrokeIndex: -1,
            ));
          }
        }
      }

      if (cubics.isEmpty) return null;

      final bounds = _computeRawBounds(const [], cubics);
      final glyph = GlyphData(cubics: cubics, bounds: bounds);
      _glyphCache[codeUnit] = glyph;
      return glyph;
    } catch (_) {
      return null;
    }
  }

  void _recomputeTimingForAnimStrokes() {
    if (_animStrokes.isEmpty) return;

    if (_animIsText) {
      for (final s in _animStrokes) {
        final curvature = s.curvatureMetricDeg;
        final double curvNorm = (curvature / 70.0).clamp(0.0, 1.0).toDouble();
        final base = _textStrokeBaseTimeSec * _textStrokeSlowdown;
        final extra = base * _textStrokeCurveExtraFrac * curvNorm;
        s.drawTimeSec = base + extra;
        s.travelTimeBeforeSec = 0.0;
        s.timeWeight = s.drawTimeSec;
        s.animationStartSec = 0.0;
        s.animationEndSec = s.drawTimeSec;
        s.animationWorkerIndex = 0;
      }
    } else {
      for (final s in _animStrokes) {
        final length = s.lengthPx;
        final curvature = s.curvatureMetricDeg;

        final lengthK = length / 1000.0;
        final double curvNorm = (curvature / 70.0).clamp(0.0, 1.0).toDouble();

        final rawTime = _minStrokeTimeSec +
            lengthK * _lengthTimePerKPxSec +
            curvNorm * _curvatureExtraMaxSec;

        s.drawTimeSec =
            rawTime.clamp(_minStrokeTimeSec, _maxStrokeTimeSec).toDouble();
        s.animationStartSec = 0.0;
        s.animationEndSec = s.drawTimeSec;
        s.animationWorkerIndex = 0;
      }

      DrawableStroke? prev;
      for (final s in _animStrokes) {
        double travel = 0.0;

        if (prev != null) {
          final lastP = prev.points.last;
          final firstP = s.points.first;
          final dist = (firstP - lastP).distance;
          final distK = dist / 1000.0;

          final rawTravel = _baseTravelTimeSec + distK * _travelTimePerKPxSec;

          travel =
              rawTravel.clamp(_minTravelTimeSec, _maxTravelTimeSec).toDouble();
        }

        s.travelTimeBeforeSec = travel;
        s.timeWeight = s.travelTimeBeforeSec + s.drawTimeSec;
        prev = s;
      }
    }

    final totalNominalSeconds =
        _animStrokes.fold<double>(0.0, (sum, d) => sum + d.timeWeight);
    final workerCount = _determineAnimationWorkerCount(
      strokes: _animStrokes,
      totalNominalSeconds: totalNominalSeconds,
    );
    final scheduledDurationSec = _scheduleAnimationAcrossWorkers(
      _animStrokes,
      workerCount: workerCount,
      maxDurationSec: _forceMaxObjectAnimationDuration
          ? _forcedMaxObjectAnimationDurationSec
          : null,
    );
    _activeAnimationWorkerCount = workerCount;
    _activeAnimationDurationSec = scheduledDurationSec;

    final animSeconds = (scheduledDurationSec > 0)
        ? (scheduledDurationSec / _globalSpeedMultiplier)
        : 0.0;

    final pending = _animationCompletionCompleter;
    if (pending != null && !pending.isCompleted) {
      pending.complete();
    }
    _animationCompletionCompleter = Completer<void>();

    if (animSeconds <= 0.0) {
      _controller.stop();
      setState(() {
        _animValue = 1.0;
        _drawableStrokes = [..._staticStrokes, ..._animStrokes];
        _status =
            'Total strokes: ${_drawableStrokes.length} | nothing to animate';
      });
      final completer = _animationCompletionCompleter;
      _animationCompletionCompleter = null;
      if (completer != null && !completer.isCompleted) {
        completer.complete();
      }
      return;
    }

    final ms = math.max(1, (animSeconds * 1000).round());
    _controller.duration = Duration(milliseconds: ms);

    _controller
      ..reset()
      ..forward();

    setState(() {
      _animValue = 0.0;
      _stepMode = false;
      _stepStrokeCount = 0;
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];
      _status =
          'Total strokes: ${_drawableStrokes.length} | anim=${ms}ms | workers=$_activeAnimationWorkerCount';
    });
  }

  int _determineAnimationWorkerCount({
    required List<DrawableStroke> strokes,
    required double totalNominalSeconds,
  }) {
    if (strokes.isEmpty || !_enableParallelStrokeWorkers) {
      return 1;
    }

    int workerCount = 1;
    if (_forceMaxObjectAnimationDuration &&
        _forcedMaxObjectAnimationDurationSec > 0) {
      workerCount = math.max(
        1,
        (totalNominalSeconds / _forcedMaxObjectAnimationDurationSec).ceil(),
      );
    }

    workerCount = workerCount.clamp(1, _maxParallelStrokeWorkers).toInt();
    workerCount = math.min(workerCount, strokes.length);
    return workerCount;
  }

  double _scheduleAnimationAcrossWorkers(
    List<DrawableStroke> strokes, {
    required int workerCount,
    double? maxDurationSec,
  }) {
    if (strokes.isEmpty) {
      return 0.0;
    }

    final safeWorkerCount = math.max(1, math.min(workerCount, strokes.length));
    final workerAvailableAt = List<double>.filled(safeWorkerCount, 0.0);
    double makespan = 0.0;

    for (final stroke in strokes) {
      int selectedWorker = 0;
      double earliestAvailable = workerAvailableAt[0];
      for (int i = 1; i < workerAvailableAt.length; i++) {
        if (workerAvailableAt[i] < earliestAvailable) {
          earliestAvailable = workerAvailableAt[i];
          selectedWorker = i;
        }
      }

      final start = earliestAvailable;
      final end = start + stroke.drawTimeSec;
      stroke.travelTimeBeforeSec = 0.0;
      stroke.timeWeight = stroke.drawTimeSec;
      stroke.animationStartSec = start;
      stroke.animationEndSec = end;
      stroke.animationWorkerIndex = selectedWorker;
      workerAvailableAt[selectedWorker] = end;
      if (end > makespan) {
        makespan = end;
      }
    }

    final cap = maxDurationSec;
    if (cap != null && cap > 0.0 && makespan > cap) {
      final scale = cap / makespan;
      makespan = 0.0;
      for (final stroke in strokes) {
        stroke.drawTimeSec *= scale;
        stroke.timeWeight = stroke.drawTimeSec;
        stroke.animationStartSec *= scale;
        stroke.animationEndSec *= scale;
        if (stroke.animationEndSec > makespan) {
          makespan = stroke.animationEndSec;
        }
      }
    }

    return makespan;
  }

  void _rebuildDrawableFromCurrentStrokes() {
    if (_animStrokes.isEmpty) return;
    _recomputeTimingForAnimStrokes();
  }

  void _restartAnimationMode() {
    if (_animStrokes.isEmpty) {
      setState(() {
        _status = 'No active animation – add an object to animate.';
      });
      return;
    }
    setState(() {
      _stepMode = false;
      _animValue = 0.0;
    });
    _controller
      ..reset()
      ..forward();
  }

  void _stepNextStroke() {
    if (_drawableStrokes.isEmpty) return;
    setState(() {
      _stepMode = true;
      _controller.stop();
      if (_stepStrokeCount < _drawableStrokes.length) {
        _stepStrokeCount += 1;
      }
    });
  }

  Future<void> _writeTextFromUi() async {
    final prompt = _textPromptController.text;
    if (prompt.isEmpty) {
      setState(() {
        _status = 'Text prompt is empty.';
      });
      return;
    }

    final x = double.tryParse(_textXController.text.trim()) ?? 0.0;
    final y = double.tryParse(_textYController.text.trim()) ?? 0.0;
    final size =
        double.tryParse(_textSizeController.text.trim()) ?? _textBaseFontSizeRef;

    final gapParsed =
        double.tryParse(_textGapController.text.trim()) ?? _textLetterGapPx;
    _textLetterGapPx = math.max(0.0, gapParsed);

    await _whiteboardActions.writeText(
      prompt: prompt,
      origin: Offset(x, y),
      letterSize: size,
      boardObjectId: prompt,
      logicalName: prompt,
      aliases: <String>{prompt},
    );
  }

  Future<void> _writeTextPrompt({
    required String prompt,
    required Offset origin,
    required double letterSize,
    double? strokeSlowdown,
    String? boardObjectId,
    String? logicalName,
    String? attachedToObjectId,
    Set<String>? aliases,
    bool isTemporary = false,
    String? ownerDiagramObjectId,
    bool syncWithBackend = true,
  }) async {
    await _writeTextPromptLocal(
      prompt: prompt,
      origin: origin,
      letterSize: letterSize,
      strokeSlowdown: strokeSlowdown,
      boardObjectId: boardObjectId,
      logicalName: logicalName,
      attachedToObjectId: attachedToObjectId,
      aliases: aliases,
      isTemporary: isTemporary,
      ownerDiagramObjectId: ownerDiagramObjectId,
      syncWithBackend: syncWithBackend,
    );

    if (_backendEnabled && syncWithBackend) {
      () async {
        try {
          await _syncCreateTextOnBackend(
            prompt: prompt,
            origin: origin,
            letterSize: letterSize,
            letterGap: _textLetterGapPx,
          );
        } catch (e) {
          if (!mounted) return;
          setState(() {
            _status += '\n[Backend text sync failed: $e]';
          });
        }
      }();
    }
  }

  Future<void> _writeTextPromptLocal({
    required String prompt,
    required Offset origin,
    required double letterSize,
    double? strokeSlowdown,
    String? boardObjectId,
    String? logicalName,
    String? attachedToObjectId,
    Set<String>? aliases,
    bool isTemporary = false,
    String? ownerDiagramObjectId,
    bool syncWithBackend = true,
  }) async {
    if (prompt.isEmpty) {
      setState(() {
        _status = 'Text prompt is empty.';
      });
      return;
    }

    if (letterSize <= 0) {
      letterSize = _textBaseFontSizeRef;
    }
    final effectiveStrokeSlowdown =
        (strokeSlowdown ?? _defaultTextStrokeSlowdown)
            .clamp(0.1, 100.0)
            .toDouble();

    await _ensureFontMetricsLoaded();
    final lineHeight = _fontLineHeightPx ?? _targetResolution * 0.5;
    final imageHeight = _fontImageHeightPx ?? _targetResolution;
    final scale = letterSize / lineHeight;

    if (_animStrokes.isNotEmpty) {
      _controller.stop();
      _staticStrokes = [..._staticStrokes, ..._animStrokes];
      _animStrokes = const [];
      _animValue = 0.0;
    }

    final newStrokes = <DrawableStroke>[];
    final diagBoard =
        math.sqrt(_boardWidth * _boardWidth + _boardHeight * _boardHeight);

    final objectId =
        (boardObjectId?.trim().isNotEmpty == true) ? boardObjectId!.trim() : prompt;
    final displayName =
        (logicalName?.trim().isNotEmpty == true) ? logicalName!.trim() : prompt;

    await _deleteObject(id: objectId, silentIfMissing: true, animate: false);

    double cursorX = origin.dx;
    final baselineWorldY = origin.dy;
    final baselineGlyph = imageHeight / 2.0;
    final baselineGlyphScaled = baselineGlyph * scale;

    final letterGapPx = _textLetterGapPx;
    const spaceWidthFactor = 0.5;

    for (int i = 0; i < prompt.length; i++) {
      final ch = prompt[i];
      final code = ch.codeUnitAt(0);

      if (ch == ' ') {
        cursorX += letterSize * spaceWidthFactor;
        continue;
      }

      final glyph = await _getGlyphForCode(code);
      if (glyph == null) {
        cursorX += letterSize * spaceWidthFactor;
        continue;
      }

      final gb = glyph.bounds;
      final glyphWidth = math.max(gb.width, 1e-3);
      final glyphLeft = gb.left;

      final letterOffsetX = cursorX - glyphLeft * scale;
      final letterOffsetY = baselineWorldY - baselineGlyphScaled;

      for (final stroke in glyph.cubics) {
        final ptsRaw = _sampleCubicStroke(stroke, upscale: scale);
        if (ptsRaw.length < 2) continue;

        final ptsPlaced = ptsRaw
            .map((p) => Offset(p.dx + letterOffsetX, p.dy + letterOffsetY))
            .toList(growable: false);

        newStrokes.add(
          _makeDrawableFromPoints(
            jsonName: objectId,
            objectOrigin: origin,
            objectScale: scale,
            pts: ptsPlaced,
            basePenWidth: _basePenWidthPx,
            diag: diagBoard,
            sourceStrokeIndex: -1,
            strokeWidthMultiplier: _textStrokeWidthMultiplierForScale(scale),
          ),
        );
      }

      final glyphWidthScaled = glyphWidth * scale;
      cursorX += glyphWidthScaled + letterGapPx;
    }

    if (newStrokes.isEmpty) {
      setState(() {
        _status = 'No strokes generated for text "$prompt".';
      });
      return;
    }

    setState(() {
      _animIsText = true;
      _textStrokeSlowdown = effectiveStrokeSlowdown;
      _animStrokes = newStrokes;
      _drawableStrokes = [..._staticStrokes, ..._animStrokes];

      _registerObject(
        BoardObjectRecord(
          objectId: objectId,
          kind: BoardObjectKind.text,
          displayName: displayName,
          origin: origin,
          scale: scale,
          fileName: null,
          processedId: null,
          aliases: <String>{
            displayName,
            prompt,
            ...?aliases,
          },
          linkedToObjectId: attachedToObjectId,
          ownerDiagramObjectId: ownerDiagramObjectId,
          isTemporary: isTemporary,
          syncWithBackend: syncWithBackend,
        ),
      );

      _status =
          'Writing text "$prompt" | strokes=${newStrokes.length}, letters=${prompt.length}';
    });

    _recomputeTimingForAnimStrokes();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Row(
        children: [
          Expanded(
            flex: 3,
            child: Container(
              color: Colors.white,
              child: CustomPaint(
                painter: WhiteboardPainter(
                  staticStrokes: _staticStrokes,
                  animStrokes: _animStrokes,
                  animationT: _animValue,
                  basePenWidth: _basePenWidthPx,
                  stepMode: _stepMode,
                  stepStrokeCount: _stepStrokeCount,
                  boardWidth: _boardWidth,
                  boardHeight: _boardHeight,
                  speedStartPct: _strokeSpeedStartPct,
                  speedEndPct: _strokeSpeedEndPct,
                  speedPeakMult: _strokeSpeedPeakMult,
                  speedPeakTime: _strokeSpeedPeakTime,
                  diagramStatesByObjectId: Map<String, DiagramRuntimeState>.from(
                    _diagramStatesByObjectId,
                  ),
                  highlightedClusterRefs: Set<String>.from(
                    _highlightedClusterRefs,
                  ),
                  diagramLabelsByClusterRef:
                      Map<String, DiagramPlacedLabel>.from(
                    _diagramLabelsByClusterRef,
                  ),
                  diagramConnections:
                      List<DiagramConnectionRecord>.from(_diagramConnections),
                  zoomedClusterRef: _zoomedClusterRef,
                  objectSaturationByObjectId:
                      Map<String, double>.from(_objectSaturationById),
                  diagramNonFocusSaturationByObjectId:
                      Map<String, double>.from(_diagramNonFocusSaturationByObjectId),
                  activeHighlightRefsByDiagramId: _activeHighlightRefsByDiagramId.map(
                    (key, value) => MapEntry(key, Set<String>.from(value)),
                  ),
                  zoomedDiagramObjectId: _zoomedDiagramObjectId,
                  boardZoomScale: _boardZoomScale,
                  boardZoomWorldCenter: _boardZoomWorldCenter,
                  animationDurationSec: _activeAnimationDurationSec,
                ),
              ),
            ),
          ),
          Container(
            width: 340,
            color: const Color(0xFF111111),
            padding: const EdgeInsets.all(16),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Control Panel',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 12),
                  ElevatedButton(
                    onPressed: _loadAndRender,
                    child: const Text('Load & Render (auto anim)'),
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: _restartAnimationMode,
                    child: const Text('Replay animation'),
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton(
                    onPressed: _stepNextStroke,
                    child: const Text('Step: next stroke'),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _status,
                    style: const TextStyle(color: Colors.white70),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _stepMode
                        ? 'Mode: STEP  | shown: $_stepStrokeCount'
                        : 'Mode: ANIM | t=${_animValue.toStringAsFixed(2)}',
                    style:
                        const TextStyle(color: Colors.white54, fontSize: 12),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'Legacy JSON path:\nwhiteboard_backend\\StrokeVectors\\edges_0_skeleton.json',
                    style: TextStyle(color: Colors.white38, fontSize: 11),
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Load JSON from folder',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Folder:\n$_vectorsFolder',
                    style:
                        const TextStyle(color: Colors.white38, fontSize: 10),
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _fileNameController,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'File name (e.g. processed_4.json)',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _posXController,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                          decoration: const InputDecoration(
                            labelText: 'X',
                            labelStyle: TextStyle(
                              color: Colors.white54,
                              fontSize: 11,
                            ),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 6,
                            ),
                          ),
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                            signed: true,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: TextField(
                          controller: _posYController,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                          decoration: const InputDecoration(
                            labelText: 'Y',
                            labelStyle: TextStyle(
                              color: Colors.white54,
                              fontSize: 11,
                            ),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 6,
                            ),
                          ),
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                            signed: true,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _scaleController,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Scale (1.0 = default 2k size)',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                      signed: false,
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Text writing (Font folder)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    r'Font JSON path: whiteboard_backend\Font\<hex>.json',
                    style: TextStyle(color: Colors.white38, fontSize: 10),
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _textPromptController,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Text to write',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _textXController,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                          decoration: const InputDecoration(
                            labelText: 'Text X (baseline)',
                            labelStyle: TextStyle(
                              color: Colors.white54,
                              fontSize: 11,
                            ),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 6,
                            ),
                          ),
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                            signed: true,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: TextField(
                          controller: _textYController,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                          ),
                          decoration: const InputDecoration(
                            labelText: 'Text Y (baseline)',
                            labelStyle: TextStyle(
                              color: Colors.white54,
                              fontSize: 11,
                            ),
                            filled: true,
                            fillColor: Color(0xFF222222),
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 6,
                            ),
                          ),
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                            signed: true,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _textSizeController,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Letter size (px on board)',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                      signed: false,
                    ),
                  ),
                  const SizedBox(height: 4),
                  TextField(
                    controller: _textGapController,
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                    decoration: const InputDecoration(
                      labelText: 'Letter gap (px)',
                      labelStyle:
                          TextStyle(color: Colors.white54, fontSize: 11),
                      filled: true,
                      fillColor: Color(0xFF222222),
                      border: OutlineInputBorder(),
                      isDense: true,
                      contentPadding:
                          EdgeInsets.symmetric(horizontal: 8, vertical: 6),
                    ),
                    keyboardType: const TextInputType.numberWithOptions(
                      decimal: true,
                      signed: false,
                    ),
                  ),
                  const SizedBox(height: 4),
                  ElevatedButton(
                    onPressed: _writeTextFromUi,
                    child: const Text('Write text'),
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Global speed',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Multiplier: ${_globalSpeedMultiplier.toStringAsFixed(2)}x',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _globalSpeedMultiplier,
                    min: 0.25,
                    max: 3.0,
                    divisions: 11,
                    label: _globalSpeedMultiplier.toStringAsFixed(2),
                    onChanged: (v) {
                      setState(() {
                        _globalSpeedMultiplier = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Stroke timing (per stroke)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Min stroke time: ${_minStrokeTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _minStrokeTimeSec,
                    min: 0.05,
                    max: _maxStrokeTimeSec,
                    divisions: 50,
                    label: _minStrokeTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _minStrokeTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Max stroke time: ${_maxStrokeTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _maxStrokeTimeSec,
                    min: _minStrokeTimeSec,
                    max: 0.8,
                    divisions: 70,
                    label: _maxStrokeTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _maxStrokeTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Length factor: ${_lengthTimePerKPxSec.toStringAsFixed(3)} s per 1000px',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _lengthTimePerKPxSec,
                    min: 0.0,
                    max: 0.3,
                    divisions: 30,
                    label: _lengthTimePerKPxSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _lengthTimePerKPxSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Max curvature extra: ${_curvatureExtraMaxSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _curvatureExtraMaxSec,
                    min: 0.0,
                    max: 0.3,
                    divisions: 30,
                    label: _curvatureExtraMaxSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _curvatureExtraMaxSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Curvature profile (within stroke)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Profile strength: ${_curvatureProfileFactor.toStringAsFixed(2)}x',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _curvatureProfileFactor,
                    min: 0.0,
                    max: 3.0,
                    divisions: 30,
                    label: _curvatureProfileFactor.toStringAsFixed(2),
                    onChanged: (v) {
                      setState(() {
                        _curvatureProfileFactor = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Angle scale: ${_curvatureAngleScale.toStringAsFixed(0)}°',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _curvatureAngleScale,
                    min: 20.0,
                    max: 160.0,
                    divisions: 14,
                    label: _curvatureAngleScale.toStringAsFixed(0),
                    onChanged: (v) {
                      setState(() {
                        _curvatureAngleScale = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Within-stroke speed curve',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Start ease: ${(_strokeSpeedStartPct * 100).toStringAsFixed(0)}%',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _strokeSpeedStartPct,
                    min: 0.0,
                    max: 0.35,
                    divisions: 35,
                    label: (_strokeSpeedStartPct * 100).toStringAsFixed(0),
                    onChanged: (v) {
                      setState(() {
                        _strokeSpeedStartPct = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _restartAnimationMode();
                      }
                    },
                  ),
                  Text(
                    'End ease: ${(_strokeSpeedEndPct * 100).toStringAsFixed(0)}%',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _strokeSpeedEndPct,
                    min: 0.0,
                    max: 0.35,
                    divisions: 35,
                    label: (_strokeSpeedEndPct * 100).toStringAsFixed(0),
                    onChanged: (v) {
                      setState(() {
                        _strokeSpeedEndPct = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _restartAnimationMode();
                      }
                    },
                  ),
                  Text(
                    'Peak speed: ${_strokeSpeedPeakMult.toStringAsFixed(2)}x',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _strokeSpeedPeakMult,
                    min: 1.0,
                    max: 4.0,
                    divisions: 30,
                    label: _strokeSpeedPeakMult.toStringAsFixed(2),
                    onChanged: (v) {
                      setState(() {
                        _strokeSpeedPeakMult = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _restartAnimationMode();
                      }
                    },
                  ),
                  Text(
                    'Peak time: ${_strokeSpeedPeakTime.toStringAsFixed(2)}',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _strokeSpeedPeakTime,
                    min: 0.0,
                    max: 1.0,
                    divisions: 40,
                    label: _strokeSpeedPeakTime.toStringAsFixed(2),
                    onChanged: (v) {
                      setState(() {
                        _strokeSpeedPeakTime = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty) {
                        _restartAnimationMode();
                      }
                    },
                  ),
                  const Text(
                    'Travel between strokes (non-text)',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Base travel: ${_baseTravelTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _baseTravelTimeSec,
                    min: 0.0,
                    max: 0.6,
                    divisions: 60,
                    label: _baseTravelTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _baseTravelTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Distance factor: ${_travelTimePerKPxSec.toStringAsFixed(3)} s per 1000px',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _travelTimePerKPxSec,
                    min: 0.0,
                    max: 0.4,
                    divisions: 40,
                    label: _travelTimePerKPxSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _travelTimePerKPxSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Travel clamp min: ${_minTravelTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _minTravelTimeSec,
                    min: 0.0,
                    max: _maxTravelTimeSec,
                    divisions: 60,
                    label: _minTravelTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _minTravelTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  Text(
                    'Travel clamp max: ${_maxTravelTimeSec.toStringAsFixed(3)} s',
                    style:
                        const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                  Slider(
                    value: _maxTravelTimeSec,
                    min: _minTravelTimeSec,
                    max: 0.8,
                    divisions: 80,
                    label: _maxTravelTimeSec.toStringAsFixed(3),
                    onChanged: (v) {
                      setState(() {
                        _maxTravelTimeSec = v;
                      });
                    },
                    onChangeEnd: (_) {
                      if (_animStrokes.isNotEmpty && !_animIsText) {
                        _rebuildDrawableFromCurrentStrokes();
                      }
                    },
                  ),
                  const SizedBox(height: 16),
                  const Divider(color: Colors.white24),
                  const SizedBox(height: 8),
                  const Text(
                    'Erase objects / text prompts',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 4),
                  if (_drawnJsonNames.isEmpty)
                    const Text(
                      'No objects drawn yet.',
                      style: TextStyle(color: Colors.white54, fontSize: 11),
                    )
                  else ...[
                    DropdownButton<String>(
                      dropdownColor: const Color(0xFF222222),
                      value: _selectedEraseName,
                      items: _drawnJsonNames
                          .map(
                            (name) => DropdownMenuItem(
                              value: name,
                              child: Text(
                                name,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                          )
                          .toList(),
                      onChanged: (v) {
                        setState(() {
                          _selectedEraseName = v;
                        });
                      },
                    ),
                    const SizedBox(height: 4),
                    ElevatedButton(
                      onPressed: _selectedEraseName == null
                          ? null
                          : () => _whiteboardActions.deleteImage(
                                name: _selectedEraseName!,
                              ),
                      child: const Text('Erase selected'),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Rect _computeRawBounds(
    List<StrokePolyline> polys,
    List<StrokeCubic> cubics,
  ) {
    double minX = double.infinity, minY = double.infinity;
    double maxX = -double.infinity, maxY = -double.infinity;

    for (final s in polys) {
      for (final p in s.points) {
        if (p.dx < minX) minX = p.dx;
        if (p.dy < minY) minY = p.dy;
        if (p.dx > maxX) maxX = p.dx;
        if (p.dy > maxY) maxY = p.dy;
      }
    }

    for (final s in cubics) {
      for (final seg in s.segments) {
        for (final p in [seg.p0, seg.c1, seg.c2, seg.p1]) {
          if (p.dx < minX) minX = p.dx;
          if (p.dy < minY) minY = p.dy;
          if (p.dx > maxX) maxX = p.dx;
          if (p.dy > maxY) maxY = p.dy;
        }
      }
    }

    if (minX == double.infinity) {
      return const Rect.fromLTWH(0, 0, 1, 1);
    }
    final w = math.max(1e-3, maxX - minX);
    final h = math.max(1e-3, maxY - minY);
    return Rect.fromLTWH(minX, minY, w, h);
  }

  List<Offset> _downsamplePolyline(List<Offset> pts, int maxPoints) {
    final n = pts.length;
    if (n <= maxPoints || maxPoints <= 2) return pts;

    double totalLen = 0.0;
    final segLen = List<double>.filled(n, 0.0);

    for (int i = 1; i < n; i++) {
      final d = (pts[i] - pts[i - 1]).distance;
      segLen[i] = d;
      totalLen += d;
    }

    if (totalLen <= 1e-6) {
      return <Offset>[pts.first, pts.last];
    }

    final out = <Offset>[];
    out.add(pts.first);

    final step = totalLen / (maxPoints - 1);
    double accum = 0.0;
    double nextTarget = step;
    int i = 1;

    while (i < n - 1 && out.length < maxPoints - 1) {
      final d = segLen[i];
      if (d <= 0.0) {
        i++;
        continue;
      }

      if (accum + d >= nextTarget) {
        final t = (nextTarget - accum) / d;
        final p = Offset(
          pts[i - 1].dx + (pts[i].dx - pts[i - 1].dx) * t,
          pts[i - 1].dy + (pts[i].dy - pts[i - 1].dy) * t,
        );
        out.add(p);
        nextTarget += step;
      } else {
        accum += d;
        i++;
      }
    }

    out.add(pts.last);
    return out;
  }

  List<DrawableStroke> _buildDrawableStrokesForObject({
    required String jsonName,
    required Offset origin,
    required double objectScale,
    required List<StrokePolyline> polylines,
    required List<StrokeCubic> cubics,
    required double srcWidth,
    required double srcHeight,
    required double targetResolution,
    required double basePenWidth,
  }) {
    final strokes = <DrawableStroke>[];

    final srcMax = math.max(srcWidth, srcHeight);
    final baseUpscale = srcMax > 0 ? targetResolution / srcMax : 1.0;
    final scale = objectScale <= 0 ? 1.0 : objectScale;
    final upscale = baseUpscale * scale;

    final diag = math.sqrt(
      math.pow(srcWidth * upscale, 2) + math.pow(srcHeight * upscale, 2),
    );
    final diagSafe = diag > 1e-3 ? diag : 1.0;

    final srcBounds = _computeRawBounds(polylines, cubics);
    final srcCenter = Offset(
      srcBounds.left + srcBounds.width / 2.0,
      srcBounds.top + srcBounds.height / 2.0,
    );
    final centerScaled = Offset(srcCenter.dx * upscale, srcCenter.dy * upscale);

    for (final s in polylines) {
      final pts = s.points
          .map((p) {
            final scaled = Offset(p.dx * upscale, p.dy * upscale);
            return Offset(
              scaled.dx - centerScaled.dx + origin.dx,
              scaled.dy - centerScaled.dy + origin.dy,
            );
          })
          .toList(growable: false);

      if (pts.length < 2) continue;

      strokes.add(_makeDrawableFromPoints(
        jsonName: jsonName,
        objectOrigin: origin,
        objectScale: scale,
        pts: pts,
        basePenWidth: basePenWidth,
        diag: diagSafe,
        strokeColor: s.color,
        sourceStrokeIndex: s.sourceStrokeIndex,
      ));
    }

    for (final c in cubics) {
      final ptsRaw = _sampleCubicStroke(c, upscale: upscale);
      final pts = ptsRaw
          .map((p) => Offset(
                p.dx - centerScaled.dx + origin.dx,
                p.dy - centerScaled.dy + origin.dy,
              ))
          .toList(growable: false);

      if (pts.length < 2) continue;

      strokes.add(_makeDrawableFromPoints(
        jsonName: jsonName,
        objectOrigin: origin,
        objectScale: scale,
        pts: pts,
        basePenWidth: basePenWidth,
        diag: diagSafe,
        strokeColor: c.color,
        sourceStrokeIndex: c.sourceStrokeIndex,
      ));
    }

    return strokes;
  }

  DrawableStroke _makeDrawableFromPoints({
    required String jsonName,
    required Offset objectOrigin,
    required double objectScale,
    required List<Offset> pts,
    required double basePenWidth,
    required double diag,
    required int sourceStrokeIndex,
    Color? strokeColor,
    double strokeWidthMultiplier = 1.0,
  }) {
    final scale = objectScale <= 0 ? 1.0 : objectScale;

    final double clampedScale = scale.clamp(0.1, 3.0).toDouble();
    double scaleFactor;
    if (clampedScale <= 1.0) {
      scaleFactor = clampedScale;
    } else {
      scaleFactor = 1.0 + 0.4 * (clampedScale - 1.0);
    }

    int effectiveMax = (_maxDisplayPointsPerStroke * scaleFactor).round();
    if (effectiveMax < 8) effectiveMax = 8;
    if (effectiveMax > _maxDisplayPointsPerStroke) {
      effectiveMax = _maxDisplayPointsPerStroke;
    }

    final workPts = _downsamplePolyline(pts, effectiveMax);
    final n = workPts.length;

    final cumGeom = List<double>.filled(n, 0.0);
    final cumCost = List<double>.filled(n, 0.0);

    double length = 0.0;
    double cost = 0.0;

    double prevSharpNorm = 0.0;
    final angleScale =
        _curvatureAngleScale.abs() < 1e-3 ? 1.0 : _curvatureAngleScale;

    for (int i = 1; i < n; i++) {
      final v = workPts[i] - workPts[i - 1];
      final segLen = v.distance;
      if (segLen < 1e-6) {
        cumGeom[i] = length;
        cumCost[i] = cost;
        continue;
      }

      length += segLen;

      double angDeg = 0.0;
      if (i > 1) {
        final vPrev = workPts[i - 1] - workPts[i - 2];
        final lenPrev = vPrev.distance;
        if (lenPrev >= 1e-6) {
          final dot = (vPrev.dx * v.dx + vPrev.dy * v.dy) / (lenPrev * segLen);
          final clamped = dot.clamp(-1.0, 1.0);
          angDeg = math.acos(clamped) * 180.0 / math.pi;
        }
      }

      final double sharpNorm = (angDeg / angleScale).clamp(0.0, 1.5).toDouble();
      final smoothedSharp = 0.7 * prevSharpNorm + 0.3 * sharpNorm;
      prevSharpNorm = smoothedSharp;
      final slowFactor = 1.0 + _curvatureProfileFactor * smoothedSharp;
      final segCost = segLen * slowFactor;

      cost += segCost;
      cumGeom[i] = length;
      cumCost[i] = cost;
    }

    double drawCostTotal;
    if (cost <= 0.0) {
      if (n > 1) {
        for (int i = 1; i < n; i++) {
          final t = i / (n - 1);
          cumGeom[i] = length * t;
          cumCost[i] = t;
        }
      }
      drawCostTotal = 1.0;
    } else {
      drawCostTotal = cost;
    }

    final centroid = _centroid(workPts);
    final curvature = _estimateCurvatureDeg(workPts);
    final bounds = _boundsOfPoints(workPts);

    double amp = 0.0;
    if (length > 0.02 * diag) {
      final double lenNorm = (length / diag).clamp(0.0, 1.0).toDouble();
      final double curvNorm = (curvature / 70.0).clamp(0.0, 1.0).toDouble();
      final baseAmp = basePenWidth * 0.9;
      amp = baseAmp *
          (0.5 + 0.8 * math.pow(lenNorm, 0.7)) *
          (0.6 + 0.4 * (1.0 - curvNorm));
      amp = amp.clamp(0.5, basePenWidth * 2.0).toDouble();
    }

    final displayPts = amp > 0.0 ? _applyWobble(workPts, amp) : workPts;

    final lengthK = length / 1000.0;
    final double curvNormGlobal = (curvature / 70.0).clamp(0.0, 1.0).toDouble();
    final rawTime = _minStrokeTimeSec +
        lengthK * _lengthTimePerKPxSec +
        curvNormGlobal * _curvatureExtraMaxSec;
    final drawTimeSec =
        rawTime.clamp(_minStrokeTimeSec, _maxStrokeTimeSec).toDouble();

    return DrawableStroke(
      jsonName: jsonName,
      objectOrigin: objectOrigin,
      objectScale: objectScale,
      points: displayPts,
      originalPoints: workPts,
      lengthPx: length,
      centroid: centroid,
      bounds: bounds,
      curvatureMetricDeg: curvature,
      cumGeomLen: cumGeom,
      cumDrawCost: cumCost,
      drawCostTotal: drawCostTotal,
      drawTimeSec: drawTimeSec,
      color: strokeColor ?? Colors.black,
      sourceStrokeIndex: sourceStrokeIndex,
      strokeWidthMultiplier: strokeWidthMultiplier,
    );
  }

  double _textStrokeWidthMultiplierForScale(double scale) {
    final safeScale = scale <= 0 ? 1.0 : scale;
    final distanceFromDefault = (safeScale - 1.0).abs();
    return (1.0 / (1.0 + 0.55 * distanceFromDefault))
        .clamp(0.45, 1.0)
        .toDouble();
  }

  Offset _centroid(List<Offset> pts) {
    if (pts.isEmpty) return Offset.zero;
    double sx = 0.0, sy = 0.0;
    for (final p in pts) {
      sx += p.dx;
      sy += p.dy;
    }
    final n = pts.length.toDouble();
    return Offset(sx / n, sy / n);
  }

  double _estimateCurvatureDeg(List<Offset> pts) {
    if (pts.length < 3) return 0.0;
    double sumAng = 0.0;
    int cnt = 0;
    for (int i = 1; i < pts.length - 1; i++) {
      final a = pts[i - 1];
      final b = pts[i];
      final c = pts[i + 1];
      final v1 = b - a;
      final v2 = c - b;
      final len1 = v1.distance;
      final len2 = v2.distance;
      if (len1 < 1e-3 || len2 < 1e-3) continue;
      final dot = (v1.dx * v2.dx + v1.dy * v2.dy) / (len1 * len2);
      final clamped = dot.clamp(-1.0, 1.0);
      final ang = math.acos(clamped) * 180.0 / math.pi;
      sumAng += ang.abs();
      cnt++;
    }
    if (cnt == 0) return 0.0;
    return sumAng / cnt;
  }

  List<Offset> _applyWobble(List<Offset> pts, double amp) {
    final n = pts.length;
    if (n < 3) return pts;

    final out = <Offset>[];
    for (int i = 0; i < n; i++) {
      final pPrev = pts[i == 0 ? 0 : i - 1];
      final pNext = pts[i == n - 1 ? n - 1 : i + 1];
      final dir = pNext - pPrev;
      final len = dir.distance;
      if (len < 1e-6) {
        out.add(pts[i]);
        continue;
      }
      final nx = -dir.dy / len;
      final ny = dir.dx / len;

      final t = i / (n - 1);
      final fade = math.sin(t * math.pi);
      final waveFast = math.sin(t * 5.0 * math.pi);
      final waveSlow = math.sin(t * 2.0 * math.pi);
      final combined = 0.6 * waveFast + 0.4 * waveSlow;
      final w = combined * amp * fade;

      final dx = nx * w;
      final dy = ny * w;
      out.add(Offset(pts[i].dx + dx, pts[i].dy + dy));
    }
    return out;
  }

  Rect _boundsOfPoints(List<Offset> pts) {
    double minX = double.infinity, minY = double.infinity;
    double maxX = -double.infinity, maxY = -double.infinity;
    for (final p in pts) {
      if (p.dx < minX) minX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy > maxY) maxY = p.dy;
    }
    if (minX == double.infinity) {
      return const Rect.fromLTWH(0, 0, 1, 1);
    }
    return Rect.fromLTRB(minX, minY, maxX, maxY);
  }

  List<Offset> _sampleCubicStroke(StrokeCubic s, {required double upscale}) {
    const stepsPerSegment = 18;
    final pts = <Offset>[];
    bool first = true;

    for (final seg in s.segments) {
      for (int i = 0; i <= stepsPerSegment; i++) {
        final t = i / stepsPerSegment;
        final p = _evalCubic(seg, t);
        final q = Offset(p.dx * upscale, p.dy * upscale);
        if (!first) {
          if ((q - pts.last).distance < 0.05) continue;
        }
        pts.add(q);
        first = false;
      }
    }
    return pts;
  }

  Offset _evalCubic(CubicSegment seg, double t) {
    final mt = 1.0 - t;
    final mt2 = mt * mt;
    final t2 = t * t;
    final x = mt2 * mt * seg.p0.dx +
        3 * mt2 * t * seg.c1.dx +
        3 * mt * t2 * seg.c2.dx +
        t2 * t * seg.p1.dx;
    final y = mt2 * mt * seg.p0.dy +
        3 * mt2 * t * seg.c1.dy +
        3 * mt * t2 * seg.c2.dy +
        t2 * t * seg.p1.dy;
    return Offset(x, y);
  }
}

class StrokePolyline {
  final List<Offset> points;
  final Color color;
  final int sourceStrokeIndex;

  const StrokePolyline(
    this.points, {
    this.color = Colors.black,
    this.sourceStrokeIndex = -1,
  });
}

class StrokeCubic {
  final List<CubicSegment> segments;
  final Color color;
  final int sourceStrokeIndex;

  const StrokeCubic(
    this.segments, {
    this.color = Colors.black,
    this.sourceStrokeIndex = -1,
  });
}

class CubicSegment {
  final Offset p0;
  final Offset c1;
  final Offset c2;
  final Offset p1;

  const CubicSegment({
    required this.p0,
    required this.c1,
    required this.c2,
    required this.p1,
  });
}

class GlyphData {
  final List<StrokeCubic> cubics;
  final Rect bounds;

  const GlyphData({
    required this.cubics,
    required this.bounds,
  });
}

class DrawableStroke {
  final String jsonName;
  final Offset objectOrigin;
  final double objectScale;
  final List<Offset> points;
  final List<Offset> originalPoints;
  final double lengthPx;
  final Offset centroid;
  final Rect bounds;
  final double curvatureMetricDeg;
  final Color color;
  final int sourceStrokeIndex;
  final List<double> cumGeomLen;
  final List<double> cumDrawCost;
  final double drawCostTotal;
  double drawTimeSec;
  double travelTimeBeforeSec;
  double timeWeight;
  double animationStartSec;
  double animationEndSec;
  int animationWorkerIndex;
  double strokeWidthMultiplier;

  int groupId = -1;
  int groupSize = 1;
  double importanceScore = 0.0;

  DrawableStroke({
    required this.jsonName,
    required this.objectOrigin,
    required this.objectScale,
    required this.points,
    required this.originalPoints,
    required this.lengthPx,
    required this.centroid,
    required this.bounds,
    required this.curvatureMetricDeg,
    required this.cumGeomLen,
    required this.cumDrawCost,
    required this.drawCostTotal,
    required this.drawTimeSec,
    required this.color,
    required this.sourceStrokeIndex,
    this.travelTimeBeforeSec = 0.0,
    this.timeWeight = 0.0,
    this.animationStartSec = 0.0,
    this.animationEndSec = 0.0,
    this.animationWorkerIndex = 0,
    this.strokeWidthMultiplier = 1.0,
  });

  DrawableStroke copyWith({
    String? jsonName,
    Offset? objectOrigin,
    double? objectScale,
    List<Offset>? points,
    List<Offset>? originalPoints,
    double? lengthPx,
    Offset? centroid,
    Rect? bounds,
    double? curvatureMetricDeg,
    Color? color,
    int? sourceStrokeIndex,
    List<double>? cumGeomLen,
    List<double>? cumDrawCost,
    double? drawCostTotal,
    double? drawTimeSec,
    double? travelTimeBeforeSec,
    double? timeWeight,
    double? animationStartSec,
    double? animationEndSec,
    int? animationWorkerIndex,
    double? strokeWidthMultiplier,
  }) {
    return DrawableStroke(
      jsonName: jsonName ?? this.jsonName,
      objectOrigin: objectOrigin ?? this.objectOrigin,
      objectScale: objectScale ?? this.objectScale,
      points: points ?? this.points,
      originalPoints: originalPoints ?? this.originalPoints,
      lengthPx: lengthPx ?? this.lengthPx,
      centroid: centroid ?? this.centroid,
      bounds: bounds ?? this.bounds,
      curvatureMetricDeg: curvatureMetricDeg ?? this.curvatureMetricDeg,
      cumGeomLen: cumGeomLen ?? this.cumGeomLen,
      cumDrawCost: cumDrawCost ?? this.cumDrawCost,
      drawCostTotal: drawCostTotal ?? this.drawCostTotal,
      drawTimeSec: drawTimeSec ?? this.drawTimeSec,
      color: color ?? this.color,
      sourceStrokeIndex: sourceStrokeIndex ?? this.sourceStrokeIndex,
      travelTimeBeforeSec: travelTimeBeforeSec ?? this.travelTimeBeforeSec,
      timeWeight: timeWeight ?? this.timeWeight,
      animationStartSec: animationStartSec ?? this.animationStartSec,
      animationEndSec: animationEndSec ?? this.animationEndSec,
      animationWorkerIndex: animationWorkerIndex ?? this.animationWorkerIndex,
      strokeWidthMultiplier:
          strokeWidthMultiplier ?? this.strokeWidthMultiplier,
    )
      ..groupId = groupId
      ..groupSize = groupSize
      ..importanceScore = importanceScore;
  }
}

enum BoardObjectKind { image, text }

class BoardObjectRecord {
  BoardObjectRecord({
    required this.objectId,
    required this.kind,
    required this.displayName,
    required this.origin,
    required this.scale,
    required this.fileName,
    required this.processedId,
    required Set<String> aliases,
    this.linkedToObjectId,
    this.ownerDiagramObjectId,
    this.isTemporary = false,
    this.syncWithBackend = true,
    this.sourceWidth,
    this.sourceHeight,
    this.sourceBounds,
  }) : aliases = aliases.toSet();

  final String objectId;
  final BoardObjectKind kind;
  final String displayName;
  Offset origin;
  double scale;
  final String? fileName;
  final String? processedId;
  final Set<String> aliases;
  String? linkedToObjectId;
  final String? ownerDiagramObjectId;
  final bool isTemporary;
  final bool syncWithBackend;
  final double? sourceWidth;
  final double? sourceHeight;
  final Rect? sourceBounds;

  String get backendDeleteName => fileName ?? displayName;
}

class DiagramClusterTarget {
  DiagramClusterTarget({
    required this.diagramName,
    required this.clusterName,
  });

  final String diagramName;
  final String clusterName;

  static DiagramClusterTarget? tryParse(String raw) {
    final marker = raw.indexOf(' : ');
    if (marker < 0) {
      return null;
    }
    final diagramName = raw.substring(0, marker).trim();
    final clusterName = raw.substring(marker + 3).trim();
    if (diagramName.isEmpty || clusterName.isEmpty) {
      return null;
    }
    return DiagramClusterTarget(
      diagramName: diagramName,
      clusterName: clusterName,
    );
  }
}

class DiagramClusterRecord {
  DiagramClusterRecord({
    required this.clusterRef,
    required this.diagramName,
    required this.label,
    required this.normalizedLabel,
    required Set<int> strokeIndexes,
    this.targetKey,
  }) : strokeIndexes = strokeIndexes.toSet();

  final String clusterRef;
  final String diagramName;
  final String label;
  final String normalizedLabel;
  final Set<int> strokeIndexes;
  final String? targetKey;

  bool isHighlighted = false;
  bool isZoomed = false;
  bool wasExplicitlyDrawn = false;
  String? lastWrittenLabel;
}

class DiagramRuntimeState {
  DiagramRuntimeState({
    required this.objectId,
    required this.diagramName,
    required this.processedId,
  });

  final String objectId;
  final String diagramName;
  final String processedId;
  final Map<String, DiagramClusterRecord> clustersByCanonicalRef = {};
  final Map<String, DiagramClusterRecord> _clustersByExactLabel = {};
  final Map<String, DiagramClusterRecord> _clustersByNormalizedLabel = {};
  final Set<int> drawnStrokeIndexes = <int>{};
  int labelWriteCount = 0;

  void addCluster(DiagramClusterRecord cluster) {
    clustersByCanonicalRef[cluster.clusterRef] = cluster;
    _clustersByExactLabel[cluster.label] = cluster;
    _clustersByNormalizedLabel[cluster.normalizedLabel] = cluster;
  }

  DiagramClusterRecord? resolveCluster(String actionLabel) {
    final exact = _clustersByExactLabel[actionLabel];
    if (exact != null) {
      return exact;
    }

    final normalized = actionLabel
        .trim()
        .toLowerCase()
        .replaceAll(RegExp(r'\s+'), ' ')
        .replaceAll(RegExp(r'[^a-z0-9 ]'), '');
    final normalizedHit = _clustersByNormalizedLabel[normalized];
    if (normalizedHit != null) {
      return normalizedHit;
    }

    for (final entry in _clustersByNormalizedLabel.entries) {
      if (normalized.contains(entry.key) || entry.key.contains(normalized)) {
        return entry.value;
      }
    }

    return null;
  }
}

class ResolvedDiagramCluster {
  const ResolvedDiagramCluster({
    required this.diagram,
    required this.cluster,
  });

  final DiagramRuntimeState diagram;
  final DiagramClusterRecord cluster;
}

class DiagramPlacedLabel {
  const DiagramPlacedLabel({
    required this.objectId,
    required this.clusterRef,
    required this.text,
  });

  final String objectId;
  final String clusterRef;
  final String text;
}

class DiagramConnectionRecord {
  const DiagramConnectionRecord({
    required this.connectionKey,
    required this.fromObjectId,
    required this.fromClusterRef,
    required this.toObjectId,
    required this.toClusterRef,
  });

  final String connectionKey;
  final String fromObjectId;
  final String fromClusterRef;
  final String toObjectId;
  final String toClusterRef;
}

class ActiveLessonAction {
  const ActiveLessonAction({
    required this.globalActionIndex,
    required this.type,
    this.diagramObjectId,
    this.primaryClusterRef,
    this.secondaryDiagramObjectId,
    this.secondaryClusterRef,
    this.tempObjectId,
    required this.affectedDiagramObjectIds,
  });

  final int globalActionIndex;
  final String type;
  final String? diagramObjectId;
  final String? primaryClusterRef;
  final String? secondaryDiagramObjectId;
  final String? secondaryClusterRef;
  final String? tempObjectId;
  final Set<String> affectedDiagramObjectIds;

  bool get isDiagramAction =>
      type == 'highlight_cluster' ||
      type == 'zoom_cluster' ||
      type == 'connect_cluster_to_cluster';
}

class VectorBlueprint {
  const VectorBlueprint({
    required this.fileName,
    required this.polylines,
    required this.cubics,
    required this.srcWidth,
    required this.srcHeight,
    required this.sourceBounds,
  });

  final String fileName;
  final List<StrokePolyline> polylines;
  final List<StrokeCubic> cubics;
  final double srcWidth;
  final double srcHeight;
  final Rect sourceBounds;
}

class DiagramLabelAnchor {
  const DiagramLabelAnchor({
    required this.rawText,
    required this.normalizedText,
    required this.sourceRect,
  });

  final String rawText;
  final String normalizedText;
  final Rect sourceRect;
}

class DiagramWriteTarget {
  const DiagramWriteTarget({
    required this.origin,
    required this.letterSize,
    required this.preferredRect,
  });

  final Offset origin;
  final double letterSize;
  final Rect preferredRect;
}

class WhiteboardPainter extends CustomPainter {
  final List<DrawableStroke> staticStrokes;
  final List<DrawableStroke> animStrokes;
  final double animationT;
  final double basePenWidth;
  final bool stepMode;
  final int stepStrokeCount;
  final double speedStartPct;
  final double speedEndPct;
  final double speedPeakMult;
  final double speedPeakTime;
  final double boardWidth;
  final double boardHeight;
  final Map<String, DiagramRuntimeState> diagramStatesByObjectId;
  final Set<String> highlightedClusterRefs;
  final Map<String, DiagramPlacedLabel> diagramLabelsByClusterRef;
  final List<DiagramConnectionRecord> diagramConnections;
  final String? zoomedClusterRef;
  final Map<String, double> objectSaturationByObjectId;
  final Map<String, double> diagramNonFocusSaturationByObjectId;
  final Map<String, Set<String>> activeHighlightRefsByDiagramId;
  final String? zoomedDiagramObjectId;
  final double boardZoomScale;
  final Offset boardZoomWorldCenter;
  final double animationDurationSec;

  const WhiteboardPainter({
    required this.staticStrokes,
    required this.animStrokes,
    required this.animationT,
    required this.basePenWidth,
    required this.stepMode,
    required this.stepStrokeCount,
    required this.boardWidth,
    required this.boardHeight,
    required this.speedStartPct,
    required this.speedEndPct,
    required this.speedPeakMult,
    required this.speedPeakTime,
    required this.diagramStatesByObjectId,
    required this.highlightedClusterRefs,
    required this.diagramLabelsByClusterRef,
    required this.diagramConnections,
    required this.zoomedClusterRef,
    required this.objectSaturationByObjectId,
    required this.diagramNonFocusSaturationByObjectId,
    required this.activeHighlightRefsByDiagramId,
    required this.zoomedDiagramObjectId,
    required this.boardZoomScale,
    required this.boardZoomWorldCenter,
    required this.animationDurationSec,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (staticStrokes.isEmpty && animStrokes.isEmpty) return;

    final allStrokes = <DrawableStroke>[
      ...staticStrokes,
      ...animStrokes,
    ];

    final bounds = _computeBounds(allStrokes);
    final scale = _computeUniformScale(bounds, size, padding: 80);
    final tx = (size.width - bounds.width * scale) / 2 - bounds.left * scale;
    final ty = (size.height - bounds.height * scale) / 2 - bounds.top * scale;

    canvas.save();
    canvas.translate(tx, ty);
    canvas.scale(scale);
    canvas.translate(boardZoomWorldCenter.dx, boardZoomWorldCenter.dy);
    canvas.scale(boardZoomScale);
    canvas.translate(-boardZoomWorldCenter.dx, -boardZoomWorldCenter.dy);

    if (stepMode) {
      final int count = stepStrokeCount.clamp(0, allStrokes.length).toInt();
      for (int i = 0; i < count; i++) {
        _drawStroke(canvas, allStrokes[i], 1.0, scale);
      }
    } else {
      for (final s in staticStrokes) {
        _drawStroke(canvas, s, 1.0, scale);
      }

      if (animStrokes.isNotEmpty) {
        final double clampedT = animationT.clamp(0.0, 1.0).toDouble();
        final totalDuration = animationDurationSec > 0
            ? animationDurationSec
            : animStrokes.fold<double>(
                0.0,
                (maxSoFar, stroke) =>
                    math.max(maxSoFar, stroke.animationEndSec),
              );
        final target = totalDuration > 0 ? totalDuration * clampedT : 0.0;

        for (final stroke in animStrokes) {
          final draw = stroke.drawTimeSec;
          final start = stroke.animationStartSec;
          final end = stroke.animationEndSec;
          if (draw <= 0.0 || end <= start) {
            continue;
          }

          if (target >= end) {
            _drawStroke(canvas, stroke, 1.0, scale);
            continue;
          }

          if (target <= start) {
            continue;
          }

          final local = (target - start) / draw;
          final double phase = local.clamp(0.0, 1.0).toDouble();
          if (phase > 0.0) {
            _drawStroke(canvas, stroke, phase, scale);
          }
        }
      }
    }

    canvas.restore();
  }

  void _drawStroke(
    Canvas canvas,
    DrawableStroke stroke,
    double phase,
    double viewScale,
  ) {
    final pts = stroke.points;
    if (pts.length < 2) return;

    const drawFrac = 0.8;
    final double local = phase.clamp(0.0, 1.0).toDouble();
    if (local <= 0.0) return;

    double drawPhase;
    if (local >= drawFrac) {
      drawPhase = 1.0;
    } else {
      drawPhase = local / drawFrac;
    }

    if (drawPhase <= 0.0) return;

    final n = pts.length;
    final totalCost = stroke.drawCostTotal;

    int idxMax;
    if (drawPhase >= 1.0 || totalCost <= 0.0) {
      idxMax = n - 1;
    } else {
      final warped = _warpStrokePhase(drawPhase);
      final targetCost = warped * totalCost;
      idxMax = _findIndexForCost(stroke.cumDrawCost, targetCost);

      if (idxMax < 1) idxMax = 1;
      if (idxMax >= n) idxMax = n - 1;
    }

    if (idxMax < 1) return;

    final path = Path()..moveTo(pts[0].dx, pts[0].dy);
    for (int i = 1; i <= idxMax; i++) {
      path.lineTo(pts[i].dx, pts[i].dy);
    }

    final double penW = ((basePenWidth * stroke.strokeWidthMultiplier) /
            viewScale)
        .clamp(0.35, 10.0)
        .toDouble();

    final paintLine = Paint()
      ..color = _effectiveStrokeColor(stroke)
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..strokeWidth = penW;

    canvas.drawPath(path, paintLine);
  }

  Color _effectiveStrokeColor(DrawableStroke stroke) {
    double saturation = objectSaturationByObjectId[stroke.jsonName] ?? 1.0;
    final diagram = diagramStatesByObjectId[stroke.jsonName];
    if (diagram != null) {
      final activeHighlights = activeHighlightRefsByDiagramId[diagram.objectId] ?? const <String>{};
      if (activeHighlights.isNotEmpty) {
        final isFocused = activeHighlights.any((clusterRef) {
          final cluster = diagram.clustersByCanonicalRef[clusterRef];
          if (cluster == null) {
            return false;
          }
          return cluster.strokeIndexes.contains(stroke.sourceStrokeIndex);
        });
        if (!isFocused) {
          saturation *= diagramNonFocusSaturationByObjectId[diagram.objectId] ?? 0.5;
        }
      } else if (zoomedDiagramObjectId == diagram.objectId && zoomedClusterRef != null) {
        final zoomCluster = diagram.clustersByCanonicalRef[zoomedClusterRef!];
        final isFocused = zoomCluster?.strokeIndexes.contains(stroke.sourceStrokeIndex) ?? false;
        if (!isFocused) {
          saturation *= diagramNonFocusSaturationByObjectId[diagram.objectId] ?? 0.0;
        }
      }
    }
    return _applySaturation(stroke.color, saturation.clamp(0.0, 1.0).toDouble());
  }

  Color _applySaturation(Color color, double factor) {
    final hsl = HSLColor.fromColor(color);
    return hsl.withSaturation((hsl.saturation * factor).clamp(0.0, 1.0)).toColor();
  }

  double _warpStrokePhase(double t) {
    t = t.clamp(0.0, 1.0).toDouble();

    final double start = speedStartPct.clamp(0.0, 0.49).toDouble();
    final double end = speedEndPct.clamp(0.0, 0.49).toDouble();

    final t1 = start;
    final t3 = (1.0 - end);

    if (t3 <= t1 + 1e-4) return t;

    double t2 = speedPeakTime.clamp(0.0, 1.0).toDouble();
    t2 = t2.clamp(t1 + 1e-4, t3 - 1e-4).toDouble();

    final double peak = speedPeakMult.clamp(1.0, 10.0).toDouble();

    double segFull(double vA, double vB, double l) {
      if (l <= 0.0) return 0.0;
      return l * (vA + vB) * 0.5;
    }

    double smoothInt(double x) {
      final x2 = x * x;
      final x4 = x2 * x2;
      final x5 = x4 * x;
      final x6 = x5 * x;
      return (x6 - 3.0 * x5 + 2.5 * x4);
    }

    double segPartial(double vA, double vB, double l, double x) {
      if (l <= 0.0) return 0.0;
      x = x.clamp(0.0, 1.0).toDouble();
      return l * (vA * x + (vB - vA) * smoothInt(x));
    }

    final l01 = t1;
    final l12 = t2 - t1;
    final l23 = t3 - t2;
    final l34 = 1.0 - t3;

    final total = segFull(0.0, 1.0, l01) +
        segFull(1.0, peak, l12) +
        segFull(peak, 1.0, l23) +
        segFull(1.0, 0.0, l34);

    if (total <= 1e-9) return t;

    double acc = 0.0;

    if (t < t1 && l01 > 0.0) {
      final x = t / l01;
      acc += segPartial(0.0, 1.0, l01, x);
      return (acc / total).clamp(0.0, 1.0).toDouble();
    } else {
      acc += segFull(0.0, 1.0, l01);
    }

    if (t < t2 && l12 > 0.0) {
      final x = (t - t1) / l12;
      acc += segPartial(1.0, peak, l12, x);
      return (acc / total).clamp(0.0, 1.0).toDouble();
    } else {
      acc += segFull(1.0, peak, l12);
    }

    if (t < t3 && l23 > 0.0) {
      final x = (t - t2) / l23;
      acc += segPartial(peak, 1.0, l23, x);
      return (acc / total).clamp(0.0, 1.0).toDouble();
    } else {
      acc += segFull(peak, 1.0, l23);
    }

    if (t < 1.0 && l34 > 0.0) {
      final x = (t - t3) / l34;
      acc += segPartial(1.0, 0.0, l34, x);
      return (acc / total).clamp(0.0, 1.0).toDouble();
    }

    return 1.0;
  }

  int _findIndexForCost(List<double> cumCost, double target) {
    final last = cumCost.length - 1;
    if (last <= 0) return 0;
    if (target >= cumCost[last]) return last;

    int lo = 1;
    int hi = last;
    int ans = 1;
    while (lo <= hi) {
      final mid = (lo + hi) >> 1;
      if (cumCost[mid] <= target) {
        ans = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return ans;
  }

  Iterable<DrawableStroke> _clusterStrokes(
    List<DrawableStroke> allStrokes,
    DiagramRuntimeState diagram,
    DiagramClusterRecord cluster,
  ) {
    return allStrokes.where(
      (stroke) =>
          stroke.jsonName == diagram.objectId &&
          cluster.strokeIndexes.contains(stroke.sourceStrokeIndex),
    );
  }

  Rect? _clusterBounds(
    List<DrawableStroke> allStrokes,
    DiagramRuntimeState diagram,
    DiagramClusterRecord cluster,
  ) {
    final strokes = _clusterStrokes(allStrokes, diagram, cluster).toList();
    if (strokes.isEmpty) {
      return null;
    }

    Rect bounds = strokes.first.bounds;
    for (int i = 1; i < strokes.length; i++) {
      bounds = bounds.expandToInclude(strokes[i].bounds);
    }
    return bounds;
  }

  Offset? _clusterCenter(
    List<DrawableStroke> allStrokes,
    DiagramRuntimeState diagram,
    DiagramClusterRecord cluster,
  ) {
    final bounds = _clusterBounds(allStrokes, diagram, cluster);
    if (bounds == null) {
      return null;
    }
    return Offset(bounds.center.dx, bounds.center.dy);
  }

  void _drawHighlightedClusters(
    Canvas canvas,
    List<DrawableStroke> allStrokes,
    double viewScale,
  ) {
    for (final clusterRef in highlightedClusterRefs) {
      for (final diagram in diagramStatesByObjectId.values) {
        final cluster = diagram.clustersByCanonicalRef[clusterRef];
        if (cluster == null) {
          continue;
        }

        final double penW = (basePenWidth * 2.2 / viewScale).clamp(0.8, 16.0).toDouble();
        final overlay = Paint()
          ..color = const Color(0xFFFFC107)
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.round
          ..strokeJoin = StrokeJoin.round
          ..strokeWidth = penW;

        for (final stroke in _clusterStrokes(allStrokes, diagram, cluster)) {
          final pts = stroke.points;
          if (pts.length < 2) {
            continue;
          }

          final path = Path()..moveTo(pts.first.dx, pts.first.dy);
          for (int i = 1; i < pts.length; i++) {
            path.lineTo(pts[i].dx, pts[i].dy);
          }
          canvas.drawPath(path, overlay);
        }
      }
    }

    if (zoomedClusterRef != null) {
      for (final diagram in diagramStatesByObjectId.values) {
        final cluster = diagram.clustersByCanonicalRef[zoomedClusterRef!];
        if (cluster == null) {
          continue;
        }
        final bounds = _clusterBounds(allStrokes, diagram, cluster);
        if (bounds == null) {
          continue;
        }
        final zoomPaint = Paint()
          ..color = const Color(0xFF7E57C2)
          ..style = PaintingStyle.stroke
          ..strokeWidth = (basePenWidth * 1.4 / viewScale).clamp(0.8, 12.0).toDouble();
        canvas.drawRect(bounds.inflate(20.0), zoomPaint);
      }
    }
  }

  void _drawDiagramLabels(
    Canvas canvas,
    List<DrawableStroke> allStrokes,
    double viewScale,
  ) {
    for (final entry in diagramLabelsByClusterRef.entries) {
      final label = entry.value;
      for (final diagram in diagramStatesByObjectId.values) {
        final cluster = diagram.clustersByCanonicalRef[entry.key];
        if (cluster == null) {
          continue;
        }

        final anchor = _clusterBounds(allStrokes, diagram, cluster);
        if (anchor == null) {
          continue;
        }

        final textPainter = TextPainter(
          text: TextSpan(
            text: label.text,
            style: TextStyle(
              color: const Color(0xFF1976D2),
              fontSize: 54.0 / viewScale,
              fontWeight: FontWeight.w600,
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();

        final offset = Offset(
          anchor.right + (28.0 / viewScale),
          anchor.top - (textPainter.height / 2),
        );
        textPainter.paint(canvas, offset);
      }
    }
  }

  void _drawDiagramConnections(
    Canvas canvas,
    List<DrawableStroke> allStrokes,
    double viewScale,
  ) {
    final paint = Paint()
      ..color = const Color(0xFF26A69A)
      ..style = PaintingStyle.stroke
      ..strokeWidth = (basePenWidth * 1.3 / viewScale).clamp(0.7, 10.0).toDouble();

    for (final connection in diagramConnections) {
      final fromDiagram = diagramStatesByObjectId[connection.fromObjectId];
      final toDiagram = diagramStatesByObjectId[connection.toObjectId];
      if (fromDiagram == null || toDiagram == null) {
        continue;
      }

      final fromCluster =
          fromDiagram.clustersByCanonicalRef[connection.fromClusterRef];
      final toCluster = toDiagram.clustersByCanonicalRef[connection.toClusterRef];
      if (fromCluster == null || toCluster == null) {
        continue;
      }

      final from = _clusterCenter(allStrokes, fromDiagram, fromCluster);
      final to = _clusterCenter(allStrokes, toDiagram, toCluster);
      if (from == null || to == null) {
        continue;
      }

      canvas.drawLine(from, to, paint);
    }
  }

  Rect _computeBounds(List<DrawableStroke> strokes) {
    final halfW = boardWidth / 2.0;
    final halfH = boardHeight / 2.0;
    return Rect.fromLTWH(-halfW, -halfH, boardWidth, boardHeight);
  }

  double _computeUniformScale(Rect bounds, Size size, {double padding = 10}) {
    final sx = (size.width - 2 * padding) / bounds.width;
    final sy = (size.height - 2 * padding) / bounds.height;
    final v = math.min(sx, sy);
    final fit = (v.isFinite && v > 0) ? v : 1.0;
    const shrinkFactor = 0.45;
    return fit * shrinkFactor;
  }

  @override
  bool shouldRepaint(covariant WhiteboardPainter oldDelegate) => true;
}
