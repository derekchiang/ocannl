open Base
open Ocannl
open Stdio
module Tn = Ir.Tnode
module IDX = Train.IDX
module TDSL = Operation.TDSL
module NTDSL = Operation.NTDSL
module CDSL = Train.CDSL
module Rand = Ir.Rand.Lib
module Asgns = Ir.Assignments

module type Backend = Ir.Backend_intf.Backend

let read_names () = In_channel.read_lines "bin/names.txt"

let bigrams s =
  let chars = String.to_list s in
  let front = '.' :: chars in
  let back = chars @ [ '.' ] in
  List.zip_exn front back

let get_all_bigrams () = List.(read_names () >>| bigrams |> concat)
let letters = List.init 26 ~f:(fun i -> Char.of_int_exn (Char.to_int 'a' + i))
let letters_with_dot = '.' :: letters

let char_to_index_tbl =
  let tbl = Hashtbl.create (module Char) in
  List.iteri letters_with_dot ~f:(fun i c -> Hashtbl.set tbl ~key:c ~data:i);
  tbl

let char_index c =
  match Hashtbl.find char_to_index_tbl c with
  | Some i -> i
  | None -> failwith (Printf.sprintf "Character not found: %c" c)

let bigrams_to_indices bigrams = List.(bigrams >>| fun (c1, c2) -> (char_index c1, char_index c2))
let print_tensor t = Tensor.print ~here:[%here] ~with_code:false ~with_grad:false `Default t

let one_hot ~num_classes indices =
  let num_classes = num_classes - 1 in
  let%op classes = TDSL.range num_classes in
  print_tensor classes;
  let%op indices_expanded = indices ++ "b|1 => b|i" in
  let%op classes_expanded = classes ++ "i => b|i" in
  let%op one_hot = indices_expanded = classes_expanded in
  one_hot

let _print_range_tensor () =
  let seed = 3 in
  Rand.init seed;
  Utils.settings.fixed_state_for_init <- Some seed;

  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let upto = 5 in
  (* let num_classes = 3 in *)
  let%op tensor = TDSL.range upto in
  (* let%op classes = TDSL.range num_classes in *)

  Train.forward_and_forget (module Backend) ctx tensor;
  print_tensor tensor

(* let () = _print_range_tensor () *)

let tensor_of_int_list lst =
  let size = List.length lst in
  lst |> List.map ~f:Float.of_int |> Array.of_list
  |> Tensor.ndarray ~batch_dims:[ size ] ~output_dims:[ 1 ]

let () =
  let seed = 11 in
  Rand.init seed;
  Utils.settings.fixed_state_for_init <- Some seed;

  let module Backend = (val Backends.fresh_backend ()) in
  let stream = Backend.(new_stream @@ get_device ~ordinal:0) in
  let ctx = Backend.make_context stream in
  let bigrams = get_all_bigrams () |> bigrams_to_indices in

  let batch_size = 5 in
  let int_input, int_output = List.unzip (List.take bigrams batch_size) in

  let input_tensor = tensor_of_int_list int_input in
  let output_tensor = tensor_of_int_list int_output in

  let inputs = input_tensor |> one_hot ~num_classes:27 in
  let outputs = output_tensor |> one_hot ~num_classes:27 in
  Train.set_hosted inputs.value;

  let random_weights = Array.init 27 ~f:(fun _ -> Random.float 2.0 -. 1.0) in
  let w = TDSL.param ~values:random_weights ~output_dims:[ 27 ] "w" in
  let%op logits = w *. inputs in
  Train.set_hosted logits.value;

  let%op counts = exp logits in
  Train.set_hosted counts.value;

  let%op probs = counts /. (counts ++ "b|...->... => b|0") in
  Train.set_hosted probs.value;

  let%op output_probs = (probs *. outputs) ++ "b|...->... => b|0" in
  Train.set_hosted output_probs.value;

  let%op loss = neg (log output_probs) in
  Train.set_hosted loss.value;

  let%op total_loss = loss ++ "...|...->... => 0" in
  Train.set_hosted total_loss.value;

  Train.forward_and_forget (module Backend) ctx total_loss;
  print_tensor inputs;
  print_tensor logits;
  print_tensor counts;
  print_tensor probs;
  print_tensor output_probs;
  print_tensor loss;
  print_tensor total_loss
