import 'dart:async';
import 'dart:convert';
import 'dart:io';

typedef LessonStreamPacketHandler = void Function(LessonStreamPacket packet);
typedef LessonStreamErrorHandler = void Function(Object error, StackTrace stackTrace);

class LessonStreamPacket {
  const LessonStreamPacket({
    required this.rawType,
    required this.rawJson,
    this.actionPacket,
  });

  final String rawType;
  final Map<String, dynamic> rawJson;
  final LessonActionPacket? actionPacket;

  factory LessonStreamPacket.fromJson(Map<String, dynamic> json) {
    final rawType = (json['type'] ?? '').toString();
    LessonActionPacket? actionPacket;
    if (rawType == 'action_packet') {
      actionPacket = LessonActionPacket.fromJson(json);
    }

    return LessonStreamPacket(
      rawType: rawType,
      rawJson: json,
      actionPacket: actionPacket,
    );
  }
}

class LessonActionPacket {
  LessonActionPacket({
    required this.rawJson,
    required this.schema,
    required this.emitPhase,
    required this.emitWordIndex,
    required this.chapterIndex,
    required this.chunkIndex,
    required this.segmentIndex,
    required this.batchIndex,
    required this.eventIndex,
    required this.globalActionIndex,
    required this.actionIndexInEvent,
    required this.planner,
    required this.eventKind,
    required this.eventType,
    required this.name,
    required this.imageName,
    required this.processedId,
    required this.stepKey,
    required this.eventStartWordIndex,
    required this.eventEndWordIndex,
    required this.eventDurationSec,
    required this.action,
  });

  final Map<String, dynamic> rawJson;
  final String schema;
  final String emitPhase;
  final int emitWordIndex;
  final int chapterIndex;
  final int chunkIndex;
  final int segmentIndex;
  final int batchIndex;
  final int eventIndex;
  final int globalActionIndex;
  final int actionIndexInEvent;
  final String planner;
  final String eventKind;
  final String eventType;
  final String name;
  final String imageName;
  final String processedId;
  final String stepKey;
  final int eventStartWordIndex;
  final int eventEndWordIndex;
  final double eventDurationSec;
  final LessonBoardAction action;

  bool get isStartPhase => emitPhase == 'start';
  bool get isEndPhase => emitPhase == 'end';

  factory LessonActionPacket.fromJson(Map<String, dynamic> json) {
    final actionJson = _asMap(json['action']);
    return LessonActionPacket(
      rawJson: json,
      schema: (json['schema'] ?? '').toString(),
      emitPhase: (json['emit_phase'] ?? '').toString(),
      emitWordIndex: _asInt(json['emit_word_index']),
      chapterIndex: _asInt(json['chapter_index']),
      chunkIndex: _asInt(json['chunk_index']),
      segmentIndex: _asInt(json['segment_index']),
      batchIndex: _asInt(json['batch_index'], defaultValue: -1),
      eventIndex: _asInt(json['event_index']),
      globalActionIndex: _asInt(json['global_action_index']),
      actionIndexInEvent: _asInt(json['action_index_in_event']),
      planner: (json['planner'] ?? '').toString(),
      eventKind: (json['event_kind'] ?? '').toString(),
      eventType: (json['event_type'] ?? '').toString(),
      name: (json['name'] ?? '').toString(),
      imageName: (json['image_name'] ?? '').toString(),
      processedId: (json['processed_id'] ?? '').toString(),
      stepKey: (json['step_key'] ?? '').toString(),
      eventStartWordIndex: _asInt(json['event_start_word_index']),
      eventEndWordIndex: _asInt(json['event_end_word_index']),
      eventDurationSec: _asDouble(json['event_duration_sec']),
      action: LessonBoardAction.fromJson(actionJson),
    );
  }
}

class LessonBoardAction {
  LessonBoardAction({
    required this.rawJson,
    required this.type,
    this.target,
    this.text,
    this.x,
    this.y,
    this.scale,
    this.newX,
    this.newY,
    this.imageId,
    this.imageName,
    this.clusterName,
    this.fromCluster,
    this.toCluster,
  });

  final Map<String, dynamic> rawJson;
  final String type;
  final String? target;
  final String? text;
  final double? x;
  final double? y;
  final double? scale;
  final double? newX;
  final double? newY;
  final String? imageId;
  final String? imageName;
  final String? clusterName;
  final String? fromCluster;
  final String? toCluster;

  factory LessonBoardAction.fromJson(Map<String, dynamic> json) {
    return LessonBoardAction(
      rawJson: json,
      type: (json['type'] ?? '').toString(),
      target: _asStringOrNull(json['target']),
      text: _asStringOrNull(json['text']),
      x: _asDoubleOrNull(json['x']),
      y: _asDoubleOrNull(json['y']),
      scale: _asDoubleOrNull(json['scale']),
      newX: _asDoubleOrNull(json['new_x']),
      newY: _asDoubleOrNull(json['new_y']),
      imageId: _asStringOrNull(json['image_id']),
      imageName: _asStringOrNull(json['image_name']),
      clusterName: _asStringOrNull(json['cluster_name']),
      fromCluster: _asStringOrNull(json['from_cluster']),
      toCluster: _asStringOrNull(json['to_cluster']),
    );
  }
}

class LessonTtsStreamAccepter {
  LessonTtsStreamAccepter({
    required LessonStreamPacketHandler onPacket,
    required LessonStreamErrorHandler onError,
    this.websocketUrl = 'ws://127.0.0.1:8765',
    this.autoReconnect = false,
    this.reconnectDelay = const Duration(seconds: 2),
  })  : _onPacket = onPacket,
        _onError = onError;

  final LessonStreamPacketHandler _onPacket;
  final LessonStreamErrorHandler _onError;
  final String websocketUrl;
  final bool autoReconnect;
  final Duration reconnectDelay;

  WebSocket? _socket;
  StreamSubscription<dynamic>? _subscription;
  bool _disposed = false;
  bool _connecting = false;

  bool get isConnected => _socket != null;

  Future<void> connect() async {
    if (_disposed || _connecting || _socket != null) {
      return;
    }

    _connecting = true;
    try {
      final socket = await WebSocket.connect(websocketUrl);
      _socket = socket;
      _subscription = socket.listen(
        _handleMessage,
        onError: (Object error, StackTrace stackTrace) {
          _onError(error, stackTrace);
        },
        onDone: _handleSocketDone,
        cancelOnError: false,
      );
    } catch (error, stackTrace) {
      _onError(error, stackTrace);
      if (autoReconnect && !_disposed) {
        await Future<void>.delayed(reconnectDelay);
        await connect();
      }
    } finally {
      _connecting = false;
    }
  }

  Future<void> sendJson(Map<String, dynamic> payload) async {
    final socket = _socket;
    if (socket == null) {
      return;
    }
    socket.add(jsonEncode(payload));
  }

  Future<void> dispose() async {
    _disposed = true;
    final subscription = _subscription;
    _subscription = null;
    await subscription?.cancel();

    final socket = _socket;
    _socket = null;
    await socket?.close();
  }

  void _handleMessage(dynamic rawMessage) {
    try {
      final decoded = jsonDecode(rawMessage as String);
      if (decoded is! Map) {
        throw const FormatException('Incoming lesson packet is not a JSON object.');
      }

      final json = Map<String, dynamic>.from(decoded as Map);
      final packet = LessonStreamPacket.fromJson(json);
      _onPacket(packet);
    } catch (error, stackTrace) {
      _onError(error, stackTrace);
    }
  }

  void _handleSocketDone() {
    final oldSocket = _socket;
    _socket = null;
    oldSocket?.close();

    if (autoReconnect && !_disposed) {
      Future<void>.delayed(reconnectDelay, () async {
        if (_disposed) {
          return;
        }
        await connect();
      });
    }
  }
}

Map<String, dynamic> _asMap(dynamic value) {
  if (value is Map<String, dynamic>) {
    return value;
  }
  if (value is Map) {
    return Map<String, dynamic>.from(value);
  }
  return const <String, dynamic>{};
}

String? _asStringOrNull(dynamic value) {
  if (value == null) {
    return null;
  }
  final text = value.toString();
  return text.isEmpty ? null : text;
}

int _asInt(dynamic value, {int defaultValue = 0}) {
  if (value is int) {
    return value;
  }
  if (value is num) {
    return value.toInt();
  }
  if (value is String) {
    return int.tryParse(value) ?? defaultValue;
  }
  return defaultValue;
}

double _asDouble(dynamic value, {double defaultValue = 0.0}) {
  if (value is double) {
    return value;
  }
  if (value is num) {
    return value.toDouble();
  }
  if (value is String) {
    return double.tryParse(value) ?? defaultValue;
  }
  return defaultValue;
}

double? _asDoubleOrNull(dynamic value) {
  if (value == null) {
    return null;
  }
  return _asDouble(value);
}
