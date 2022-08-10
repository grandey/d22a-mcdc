CDF      
      time       bnds      lon       lat          -   CDI       @Climate Data Interface version 2.0.5 (https://mpimet.mpg.de/cdi)   Conventions       CF-1.7 CMIP-6.2    source       0MIROC6 (2017): 
aerosol: SPRINTARS6.0
atmos: CCSR AGCM (T85; 256 x 128 longitude/latitude; 81 levels; top level 0.004 hPa)
atmosChem: none
land: MATSIRO6.0
landIce: none
ocean: COCO4.9 (tripolar primarily 1deg; 360 x 256 longitude/latitude; 63 levels; top grid cell 0-2 m)
ocnBgchem: none
seaIce: COCO4.9   institution      QJAMSTEC (Japan Agency for Marine-Earth Science and Technology, Kanagawa 236-0001, Japan), AORI (Atmosphere and Ocean Research Institute, The University of Tokyo, Chiba 277-8564, Japan), NIES (National Institute for Environmental Studies, Ibaraki 305-8506, Japan), and R-CCS (RIKEN Center for Computational Science, Hyogo 650-0047, Japan)      activity_id       CMIP   branch_method         standard   branch_time_in_child                 branch_time_in_parent         AJ�       creation_date         2020-01-29T22:50:21Z   data_specs_version        01.00.31   
experiment        pre-industrial control     experiment_id         	piControl      forcing_index               	frequency         year   further_info_url      Ihttps://furtherinfo.es-doc.org/CMIP6.MIROC.MIROC6.piControl.none.r1i1p1f1      grid      -native ocean tripolar grid with 360x256 cells      
grid_label        gm     history      Wed Aug 10 15:22:28 2022: cdo -O -s -a -f nc -mergetime /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.piControl.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_piControl_r1i1p1f1_gm_320001-399912.1d.yearmean.fldmean.nc /Users/benjamin/github/d22a-mcdc/data/regrid_missto0_yearmean_fldmean_mergetime/zostoga/MIROC6_r1i1p1f1/zostoga_MIROC6_r1i1p1f1_piControl.mergetime.nc
Thu Apr 07 23:33:16 2022: cdo -O -s -selname,zostoga -fldmean /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.piControl.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_piControl_r1i1p1f1_gm_320001-399912.1d.yearmean.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean_fldmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.piControl.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_piControl_r1i1p1f1_gm_320001-399912.1d.yearmean.fldmean.nc
Thu Apr 07 23:33:15 2022: cdo -O -s --reduce_dim -selname,zostoga -yearmean /Users/benjamin/Data/p22b/CMIP6/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.piControl.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_piControl_r1i1p1f1_gm_320001-399912.nc /Users/benjamin/Data/p22c/CMIP6/regrid_missto0_yearmean/zostoga/MIROC6_r1i1p1f1/CMIP6.CMIP.MIROC.MIROC6.piControl.r1i1p1f1.Omon.zostoga.gm.v20200130/zostoga_Omon_MIROC6_piControl_r1i1p1f1_gm_320001-399912.1d.yearmean.nc
2020-01-29T22:50:21Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.7 CMIP-6.2 and CF standards.   initialization_index            institution_id        MIROC      mip_era       CMIP6      nominal_resolution        100 km     parent_activity_id        CMIP   parent_experiment_id      piControl-spinup   parent_mip_era        CMIP6      parent_source_id      MIROC6     parent_time_units         days since 2200-1-1    parent_variant_label      r1i1p1f1   physics_index               product       model-output   realization_index               realm         ocean      	source_id         MIROC6     source_type       	AOGCM AER      sub_experiment        none   sub_experiment_id         none   table_id      Omon   
table_info        ACreation Date:(22 July 2019) MD5:b4cefb4b6dbb146fea9677a552a00934      title          MIROC6 output prepared for CMIP6   variable_id       zostoga    variant_label         r1i1p1f1   license      !CMIP6 model data produced by MIROC is licensed under a Creative Commons Attribution ShareAlike 4.0 International License (https://creativecommons.org/licenses/). Consult https://pcmdi.llnl.gov/CMIP6/TermsOfUse for terms of use governing CMIP6 output, including citation requirements and proper acknowledgment. Further information about this data, including some limitations, can be found via the further_info_url (recorded as a global attribute in this file). The data producers and data providers make no warranty, either express or implied, including, but not limited to, warranties of merchantability and fitness for a particular purpose. All liabilities arising from the supply of the information (including any liability arising in negligence) are excluded to the fullest extent permitted by law.      cmor_version      3.5.0      tracking_id       1hdl:21.14100/259bec01-5219-4ac6-a17b-a2fc5a109305      CDO       @Climate Data Operators version 2.0.5 (https://mpimet.mpg.de/cdo)         time                standard_name         time   	long_name         time   bounds        	time_bnds      units         day as %Y%m%d.%f   calendar      	gregorian      axis      T               d   	time_bnds                                 l   lon                standard_name         	longitude      	long_name         	longitude      units         degrees_east   axis      X               T   lat                standard_name         latitude   	long_name         latitude   units         degrees_north      axis      Y               \   zostoga                    
   standard_name         ,global_average_thermosteric_sea_level_change   	long_name         ,Global Average Thermosteric Sea Level Change   units         m      
_FillValue        `�x�   missing_value         `�x�   cell_methods      area: mean where sea time: mean    comment       /There is no CMIP6 request for zosga nor zossga.    original_name         shstg      original_units        cm     history       �2020-01-29T22:50:21Z altered by CMOR: Converted units from 'cm' to 'm'. 2020-01-29T22:50:21Z altered by CMOR: replaced missing value flag (-999) and corresponding data with standard missing value (1e+20).            |                A~���   A~��P   A~��P   :�A~��   A~��P   A~�hP   :�yA~���   A~�hP   A~��P   ; �)A~���   A~��P   A~�JP   :��jA~�o�   A~�JP   A~��P   ;�A~���   A~��P   A~�,P   7��A~�Q�   A~�,P   A~��P   ����A~���   A~��P   A~�P   ���A~�3�   A~�P   A~�P   :��A~���   A~�P   A~��P   ;
�A~��   A~��P   A~�aP   ;�A~���   A~�aP   A~��P   9��A~���   A~��P   A~�CP   �V�=A~�h�   A~�CP   A~��P   ���A~���   A~��P   A~�%P   :�t�A~�J�   A~�%P   A~��P   :ա)A~���   A~��P   A~�P   ;%��A~�,�   A~�P   A~�xP   ;z7�A~���   A~�xP   A~��P   ;��;A~��   A~��P   A~�ZP   ;��A~��   A~�ZP   A~��P   ;�?6A~���   A~��P   A~�<P   ;t��A~�a�   A~�<P   A~��P   ;r�VA~���   A~��P   A~�P   ;�CA~�C�   A~�P   A~��P   ;�5�A~���   A~��P   A~� P   ;n�A~�%�   A~� P   A~�qP   ;%l�A~Ɩ�   A~�qP   A~��P   :��WA~��   A~��P   A~�SP   ;>nA~�x�   A~�SP   A~��P   ;H��A~���   A~��P   A~�5P   ;@�`A~�Z�   A~�5P   A~ҦP   ;"�A~���   A~ҦP   A~�P   :�1�A~�<�   A~�P   A~׈P   ;�PA~׭�   A~׈P   A~��P   ;!��A~��   A~��P   A~�jP   ;�_A~܏�   A~�jP   A~��P   ;�A~� �   A~��P   A~�LP   ;�\A~�q�   A~�LP   A~�P   ;AE�A~���   A~�P   A~�.P   ;h�A~�S�   A~�.P   A~�P   ;Q��A~���   A~�P   A~�P   ;[%�A~�5�   A~�P   A~�P   ;o�pA~���   A~�P   A~��P   ;��bA~��   A~��P   A~�cP   ;]�A~��   A~�cP   A~��P   :��}A~���   A~��P   A~�EP   :���A~�j�   A~�EP   A~��P   ; DA~���   A~��P   A~�'P   ;.vA~�L�   A~�'P   A~��P   ;<nLA~���   A~��P   A	P   ;9�BA.�   A	P   AzP   ;&�.A��   AzP   A�P   ;n�A�   A�P   A\P   ;x	A��   A\P   A
�P   ;"S�A
��   A
�P   A>P   ;F�lAc�   A>P   A�P   ;2bAA��   A�P   A P   ;L5AE�   A P   A�P   ;1A��   A�P   AP   ;!��A'�   AP   AsP   ;�A��   AsP   A�P   ;2A	�   A�P   AUP   ;F��Az�   AUP   A �P   ;�.A ��   A �P   A#7P   ;D A#\�   A#7P   A%�P   ;�AA%��   A%�P   A(P   :��A(>�   A(P   A*�P   ;&��A*��   A*�P   A,�P   ;"5mA- �   A,�P   A/lP   ;HȋA/��   A/lP   A1�P   ;e:�A2�   A1�P   A4NP   ;���A4s�   A4NP   A6�P   ;��1A6��   A6�P   A90P   ;��:A9U�   A90P   A;�P   ;��VA;��   A;�P   A>P   ;��A>7�   A>P   A@�P   ;���A@��   A@�P   AB�P   ;*c$AC�   AB�P   AEeP   ;EAE��   AEeP   AG�P   ;G"�AG��   AG�P   AJGP   ;Y��AJl�   AJGP   AL�P   ;u�AL��   AL�P   AO)P   ;/�IAON�   AO)P   AQ�P   ;*��AQ��   AQ�P   ATP   ;<c�AT0�   ATP   AV|P   ;S\`AV��   AV|P   AX�P   ;]��AY�   AX�P   A[^P   ;}i�A[��   A[^P   A]�P   ;���A]��   A]�P   A`@P   ;�:�A`e�   A`@P   Ab�P   ;��CAb��   Ab�P   Ae"P   ;V�3AeG�   Ae"P   Ag�P   :ӟAg��   Ag�P   AjP   :�kAj)�   AjP   AluP   ;;�Al��   AluP   An�P   ;P��Ao�   An�P   AqWP   ;�0Aq|�   AqWP   As�P   ;�5�As��   As�P   Av9P   ;�H�Av^�   Av9P   Ax�P   ;��nAx��   Ax�P   A{P   ;~\A{@�   A{P   A}�P   ;&�A}��   A}�P   A�P   ;HD�A�"�   A�P   A�nP   ;:��A���   A�nP   A��P   ;{�A��   A��P   A�PP   ;�)A�u�   A�PP   A��P   ;��fA���   A��P   A�2P   ;��A�W�   A�2P   A��P   ;�YA���   A��P   A�P   ;���A�9�   A�P   A��P   ;�aYA���   A��P   A��P   ;���A��   A��P   A�gP   ;�)�A���   A�gP   A��P   ;�jA���   A��P   A�IP   ;�R!A�n�   A�IP   A��P   ;���A���   A��P   A�+P   ;;XqA�P�   A�+P   A��P   ;`�JA���   A��P   A�P   ;���A�2�   A�P   A�~P   ;�f\A���   A�~P   A��P   ;�)�A��   A��P   A�`P   ;��JA���   A�`P   A��P   ;��A���   A��P   A�BP   ;Ձ�A�g�   A�BP   A��P   ;׽�A���   A��P   A�$P   ;�)|A�I�   A�$P   A��P   ;֏yA���   A��P   A�P   ;��&A�+�   A�P   A�wP   ;���A���   A�wP   A��P   ;�V�A��   A��P   A�YP   ;��fA�~�   A�YP   A��P   ;��&A���   A��P   A�;P   ;�S�A�`�   A�;P   AˬP   ;���A���   AˬP   A�P   ;���A�B�   A�P   AЎP   ;���Aг�   AЎP   A��P   ;ʖ�A�$�   A��P   A�pP   ;֔kAՕ�   A�pP   A��P   ;�"�A��   A��P   A�RP   ;���A�w�   A�RP   A��P   ;�<vA���   A��P   A�4P   ;�G�A�Y�   A�4P   A�P   ;��AA���   A�P   A�P   ;�-�A�;�   A�P   A�P   ;a{�A��   A�P   A��P   ;���A��   A��P   A�iP   ;�k�A��   A�iP   A��P   ;��A���   A��P   A�KP   ;��A�p�   A�KP   A�P   ;���A���   A�P   A�-P   ;�LA�R�   A�-P   A��P   ;�BA���   A��P   A�P   ;�4�A�4�   A�P   A��P   ;��A���   A��P   A��P   ;�@DA��   A��P   A� �(   ;�>jA� ��   A� �(   A��   ;���A��j   A��   A�"(   ;�O�A�4�   A�"(   A�Z�   ;�EBA�mj   A�Z�   A��(   ;a�FA���   A��(   A�˨   ;6�@A��j   A�˨   A�(   ;�q�A��   A�(   A�	<�   ;���A�	Oj   A�	<�   A�
u(   ;�r,A�
��   A�
u(   A���   ;��A��j   A���   A��(   ;���A���   A��(   A��   ;��pA�1j   A��   A�W(   ;��tA�i�   A�W(   A���   <�fA��j   A���   A��(   <QA���   A��(   A� �   <ݫA�j   A� �   A�9(   ;�)A�K�   A�9(   A�q�   ;�Z�A��j   A�q�   A��(   ;�3sA���   A��(   A��   ;���A��j   A��   A�(   ;�8A�-�   A�(   A�S�   ;��A�fj   A�S�   A��(   ;ͦ�A���   A��(   A�Ĩ   ;��A��j   A�Ĩ   A��(   ;��tA��   A��(   A�5�   ;��NA�Hj   A�5�   A� n(   ;��]A� ��   A� n(   A�!��   <��A�!�j   A�!��   A�"�(   <G�A�"��   A�"�(   A�$�   <#�A�$*j   A�$�   A�%P(   <fdA�%b�   A�%P(   A�&��   <�JA�&�j   A�&��   A�'�(   <�PA�'��   A�'�(   A�(��   <��A�)j   A�(��   A�*2(   <z�A�*D�   A�*2(   A�+j�   <_{A�+}j   A�+j�   A�,�(   <#��A�,��   A�,�(   A�-ۨ   <)iA�-�j   A�-ۨ   A�/(   <'(A�/&�   A�/(   A�0L�   <	� A�0_j   A�0L�   A�1�(   < �A�1��   A�1�(   A�2��   <)jA�2�j   A�2��   A�3�(   <�IA�4�   A�3�(   A�5.�   <\�A�5Aj   A�5.�   A�6g(   <x�A�6y�   A�6g(   A�7��   <�1A�7�j   A�7��   A�8�(   <i�A�8��   A�8�(   A�:�   <n�A�:#j   A�:�   A�;I(   <0^A�;[�   A�;I(   A�<��   ;��A�<�j   A�<��   A�=�(   ;ԍ�A�=��   A�=�(   A�>�   ;�o�A�?j   A�>�   A�@+(   ;���A�@=�   A�@+(   A�Ac�   ;�hA�Avj   A�Ac�   A�B�(   <��A�B��   A�B�(   A�CԨ   <ըA�C�j   A�CԨ   A�E(   <*H�A�E�   A�E(   A�FE�   <6z�A�FXj   A�FE�   A�G~(   <<�A�G��   A�G~(   A�H��   <C��A�H�j   A�H��   A�I�(   <I�A�J�   A�I�(   A�K'�   <?�A�K:j   A�K'�   A�L`(   <	��A�Lr�   A�L`(   A�M��   ;�{A�M�j   A�M��   A�N�(   <X3A�N��   A�N�(   A�P	�   <!A�Pj   A�P	�   A�QB(   <2��A�QT�   A�QB(   A�Rz�   <<`�A�R�j   A�Rz�   A�S�(   <>�EA�S��   A�S�(   A�T�   <-�]A�T�j   A�T�   A�V$(   <<*A�V6�   A�V$(   A�W\�   <!3=A�Woj   A�W\�   A�X�(   <9�^A�X��   A�X�(   A�Yͨ   <:��A�Y�j   A�Yͨ   A�[(   <-��A�[�   A�[(   A�\>�   <��A�\Qj   A�\>�   A�]w(   ;��A�]��   A�]w(   A�^��   <��A�^�j   A�^��   A�_�(   <&_�A�_��   A�_�(   A�a �   <8K�A�a3j   A�a �   A�bY(   <.;oA�bk�   A�bY(   A�c��   <8A�c�j   A�c��   A�d�(   <2x�A�d��   A�d�(   A�f�   <1�eA�fj   A�f�   A�g;(   <,W�A�gM�   A�g;(   A�hs�   <'�lA�h�j   A�hs�   A�i�(   <5 A�i��   A�i�(   A�j�   <>x�A�j�j   A�j�   A�l(   <:.A�l/�   A�l(   A�mU�   <4@�A�mhj   A�mU�   A�n�(   <,�A�n��   A�n�(   A�oƨ   <3	"A�o�j   A�oƨ   A�p�(   <E�aA�q�   A�p�(   A�r7�   <A~�A�rJj   A�r7�   A�sp(   <7�UA�s��   A�sp(   A�t��   <)��A�t�j   A�t��   A�u�(   <$ A�u��   A�u�(   A�w�   <7�A�w,j   A�w�   A�xR(   <;A�xd�   A�xR(   A�y��   <,�&A�y�j   A�y��   A�z�(   <:��A�z��   A�z�(   A�{��   <8��A�|j   A�{��   A�}4(   <)��A�}F�   A�}4(   A�~l�   <!�=A�~j   A�~l�   A��(   <.,A���   A��(   A��ݨ   <1\,A���j   A��ݨ   A��(   <*�A��(�   A��(   A��N�   <,�<A��aj   A��N�   A���(   <BZ�A����   A���(   A����   <1�dA���j   A����   A���(   <~�A��
�   A���(   A��0�   <%�fA��Cj   A��0�   A��i(   <6�;A��{�   A��i(   A����   <9��A���j   A����   A���(   <AO_A����   A���(   A���   <Q�A��%j   A���   A��K(   <X��A��]�   A��K(   A����   <4g_A���j   A����   A���(   <qBA����   A���(   A����   <"�A��j   A����   A��-(   <'z�A��?�   A��-(   A��e�   <F��A��xj   A��e�   A���(   <Y�A����   A���(   A��֨   <_�,A���j   A��֨   A��(   <]��A��!�   A��(   A��G�   <W��A��Zj   A��G�   A���(   <3�A����   A���(   A����   <�A���j   A����   A���(   </��A���   A���(   A��)�   <H�A��<j   A��)�   A��b(   <L�$A��t�   A��b(   A����   <D,A���j   A����   A���(   <E�A����   A���(   A���   <G�ZA��j   A���   A��D(   <$'LA��V�   A��D(   A��|�   <&�A���j   A��|�   A���(   <�A����   A���(   A����   <(��A�� j   A����   A��&(   <%�A��8�   A��&(   A��^�   <;mA��qj   A��^�   A���(   <G �A����   A���(   A��Ϩ   <R��A���j   A��Ϩ   A��(   <V(�A���   A��(   A��@�   <\j�A��Sj   A��@�   A��y(   <_kwA����   A��y(   A����   <@ԅA���j   A����   A���(   <-�A����   A���(   A��"�   <<^?A��5j   A��"�   A��[(   <PEA��m�   A��[(   A����   <ipxA���j   A����   A���(   <x�5A����   A���(   A���   <{ԙA��j   A���   A��=(   <dU�A��O�   A��=(   A��u�   <Q�mA���j   A��u�   A���(   <\ѿA����   A���(   A���   <d��A���j   A���   A��(   <Z�A��1�   A��(   A��W�   <S��A��jj   A��W�   A���(   <P��A����   A���(   A��Ȩ   <^WPA���j   A��Ȩ   A��(   <h	=A���   A��(   A��9�   <�QA��Lj   A��9�   A��r(   <���A�Ƅ�   A��r(   A�Ǫ�   <�8 A�ǽj   A�Ǫ�   A���(   <��A����   A���(   A���   <o3NA��.j   A���   A��T(   <ldA��f�   A��T(   A�̌�   <k0�A�̟j   A�̌�   A���(   <n�A����   A���(   A����   <fi�A��j   A����   A��6(   <d"hA��H�   A��6(   A��n�   <bÎA�сj   A��n�   A�ҧ(   <v��A�ҹ�   A�ҧ(   A��ߨ   <���A���j   A��ߨ   A��(   <|��A��*�   A��(   A��P�   <���A��cj   A��P�   A�׉(   <��sA�כ�   A�׉(   A����   <z�CA���j   A����   A���(   <[��A���   A���(   A��2�   <A�tA��Ej   A��2�   A��k(   <Nk�A��}�   A��k(   A�ݣ�   <i_XA�ݶj   A�ݣ�   A���(   <rJA����   A���(   A���   <}RsA��'j   A���   A��M(   <q�A��_�   A��M(   A�Ⅸ   <q��A��j   A�Ⅸ   A��(   <m��A����   A��(   A����   <U�A��	j   A����   A��/(   <>@$A��A�   A��/(   A��g�   <E(cA��zj   A��g�   A��(   <\��A���   A��(   A��ب   <l��A���j   A��ب   A��(   <j��A��#�   A��(   A��I�   <UJ�A��\j   A��I�   A��(   <M�CA���   A��(   A�   <\-~A���j   A�   A���(   <Z�A���   A���(   A��+�   <^�A��>j   A��+�   A��d(   <_��A��v�   A��d(   A��   <^�A��j   A��   A���(   <YґA����   A���(   A���   <`��A�� j   A���   A��F(   <��A��X�   A��F(   A��~�   <�[�A���j   A��~�   A���(   <�( A����   A���(   A���   <}�6A��j   A���   A��((   <v�A��:�   A��((   A��`�   <yANA��sj   A��`�   A���(   <z?�A����   A���(   A��Ѩ   <wG�A���j   A��Ѩ   A�
(   <u��A��   A�
(   A�B�   <���A�Uj   A�B�   A�{(   <�yA���   A�{(   A���   <�|�A��j   A���   A��(   <k1�A���   A��(   A�$�   <M@�A�7j   A�$�   A�](   <\��A�o�   A�](   A�	��   <i�HA�	�j   A�	��   A�
�(   <~CeA�
��   A�
�(   A��   <~��A�j   A��   A�?(   <�A�Q�   A�?(   A�w�   <�!#A��j   A�w�   A��(   <�_�A���   A��(   A��   <��cA��j   A��   A�!(   <�A�3�   A�!(   A�Y�   <��A�lj   A�Y�   A��(   <�n�A���   A��(   A�ʨ   <�A��j   A�ʨ   A�(   <�/�A��   A�(   A�;�   <���A�Nj   A�;�   A�t(   <���A���   A�t(   A���   <�e�A��j   A���   A��(   <��A���   A��(   A��   <t:�A�0j   A��   A�V(   <c��A�h�   A�V(   A���   <ce�A��j   A���   A� �(   <l*�A� ��   A� �(   A�!��   <j��A�"j   A�!��   A�#8(   <^�oA�#J�   A�#8(   A�$p�   <TwA�$�j   A�$p�   A�%�(   <`�A�%��   A�%�(   A�&�   <l��A�&�j   A�&�   A�((   <v�A�(,�   A�((   A�)R�   <{H�A�)ej   A�)R�   A�*�(   <�HA�*��   A�*�(   A�+è   <�l�A�+�j   A�+è   A�,�(   <��A�-�   A�,�(   A�.4�   <o�A�.Gj   A�.4�   A�/m(   <e�[A�/�   A�/m(   A�0��   <g��A�0�j   A�0��   A�1�(   <pGWA�1��   A�1�(   A�3�   <x A�3)j   A�3�   A�4O(   <|�A�4a�   A�4O(   A�5��   <���A�5�j   A�5��   A�6�(   <��A�6��   A�6�(   A�7��   <�XA�8j   A�7��   A�91(   <q��A�9C�   A�91(   A�:i�   <fۏA�:|j   A�:i�   A�;�(   <jt�A�;��   A�;�(   A�<ڨ   <j�A�<�j   A�<ڨ   A�>(   <s�NA�>%�   A�>(   A�?K�   <]�'A�?^j   A�?K�   A�@�(   <K��A�@��   A�@�(   A�A��   <Nf*A�A�j   A�A��   A�B�(   <k<�A�C�   A�B�(   A�D-�   <~cA�D@j   A�D-�   A�Ef(   <x��A�Ex�   A�Ef(   A�F��   <r�A�F�j   A�F��   A�G�(   <��LA�G��   A�G�(   A�I�   <�ŮA�I"j   A�I�   A�JH(   <���A�JZ�   A�JH(   A�K��   <�j!A�K�j   A�K��   A�L�(   <�ńA�L��   A�L�(   A�M�   <z��A�Nj   A�M�   A�O*(   <���A�O<�   A�O*(   A�Pb�   <�>�A�Puj   A�Pb�   A�Q�(   <xzSA�Q��   A�Q�(   A�RӨ   <q7sA�R�j   A�RӨ   A�T(   <{��A�T�   A�T(   A�UD�   <�H�A�UWj   A�UD�   A�V}(   <�!A�V��   A�V}(   A�W��   <�u�A�W�j   A�W��   A�X�(   <v�A�Y �   A�X�(   A�Z&�   <fA�Z9j   A�Z&�   A�[_(   <rTHA�[q�   A�[_(   A�\��   <��A�\�j   A�\��   A�]�(   <�/}A�]��   A�]�(   A�_�   <���A�_j   A�_�   A�`A(   <���A�`S�   A�`A(   A�ay�   <��A�a�j   A�ay�   A�b�(   <���A�b��   A�b�(   A�c�   <��A�c�j   A�c�   A�e#(   <���A�e5�   A�e#(   A�f[�   <�\HA�fnj   A�f[�   A�g�(   <�A�g��   A�g�(   A�h̨   <�X
A�h�j   A�h̨   A�j(   <��;A�j�   A�j(   A�k=�   <��VA�kPj   A�k=�   A�lv(   <�ǂA�l��   A�lv(   A�m��   <���A�m�j   A�m��   A�n�(   <���A�n��   A�n�(   A�p�   <�m�A�p2j   A�p�   A�qX(   <��2A�qj�   A�qX(   A�r��   <��A�r�j   A�r��   A�s�(   <���A�s��   A�s�(   A�u�   <�AlA�uj   A�u�   A�v:(   <�R�A�vL�   A�v:(   A�wr�   <��yA�w�j   A�wr�   A�x�(   <��A�x��   A�x�(   A�y�   <�@jA�y�j   A�y�   A�{(   <��A�{.�   A�{(   A�|T�   <��\A�|gj   A�|T�   A�}�(   <��A�}��   A�}�(   A�~Ũ   <�r*A�~�j   A�~Ũ   A��(   <���A���   A��(   A��6�   <�-�A��Ij   A��6�   A��o(   <���A����   A��o(   A����   <���A���j   A����   A���(   <�՚A����   A���(   A���   <�~A��+j   A���   A��Q(   <�_�A��c�   A��Q(   A����   <���A���j   A����   A���(   <�ŨA����   A���(   A����   <�~UA��j   A����   A��3(   <��A��E�   A��3(   A��k�   <�u\A��~j   A��k�   A���(   <���A����   A���(   A��ܨ   <�}�A���j   A��ܨ   A��(   <���A��'�   A��(   A��M�   <�+�A��`j   A��M�   A���(   <��A����   A���(   A����   <��{A���j   A����   A���(   <��jA��	�   A���(   A��/�   <���A��Bj   A��/�   A��h(   <�$A��z�   A��h(   A����   <�6�A���j   A����   A���(   <�lqA����   A���(   A���   <�bA��$j   A���   A��J(   <�G�A��\�   A��J(   A����   <��hA���j   A����   A���(   <��5A����   A���(   A���   <��>A��j   A���   A��,(   <��tA��>�   A��,(   A��d�   <�t�A��wj   A��d�   A���(   <��A����   A���(   A��ը   <�$�A���j   A��ը   A��(   <Ŧ�A�� �   A��(   A��F�   <��A��Yj   A��F�   A��(   <��SA����   A��(   A����   <��A���j   A����   A���(   <��A���   A���(   A��(�   <�=5A��;j   A��(�   A��a(   <��oA��s�   A��a(   A����   <���A���j   A����   A���(   <�A����   A���(   A��
�   <��A��j   A��
�   A��C(   <��%A��U�   A��C(   A��{�   <�ҧA���j   A��{�   A���(   <�6(A����   A���(   A���   <�3
A���j   A���   A��%(   <���A��7�   A��%(   A��]�   <��\A��pj   A��]�   A���(   <�d�A����   A���(   A��Ψ   <�%�A���j   A��Ψ   A��(   <��A���   A��(   A��?�   <�'�A��Rj   A��?�   A��x(   <��A����   A��x(   A����   <���A���j   A����   A���(   <�a�A����   A���(   A��!�   <�ǏA��4j   A��!�   A��Z(   <���A��l�   A��Z(   A�Œ�   <�c�A�ťj   A�Œ�   A���(   <���A����   A���(   A���   <��A��j   A���   A��<(   <���A��N�   A��<(   A��t�   <�Y7A�ʇj   A��t�   A�˭(   <���A�˿�   A�˭(   A���   <��&A���j   A���   A��(   <�FMA��0�   A��(   A��V�   <�B�A��ij   A��V�   A�Џ(   <��EA�С�   A�Џ(   A��Ǩ   <��IA���j   A��Ǩ   A�� (   <��mA���   A�� (   A��8�   <�a`A��Kj   A��8�   A��q(   <���A�Ճ�   A��q(   A�֩�   <�J�A�ּj   A�֩�   A���(   <���A����   A���(   A���   <�<A��-j   A���   A��S(   <��9A��e�   A��S(   A�ۋ�   <�]=A�۞j   A�ۋ�   A���(   <���A����   A���(   A����   <��qA��j   A����   A��5(   <��,A��G�   A��5(   A��m�   <�T,A���j   A��m�   A��(   <�$DA���   A��(   A��ި   <�ߋA���j   A��ި   A��(   <�NwA��)�   A��(   A��O�   <��bA��bj   A��O�   A��(   <���A���   A��(   A����   <��"A���j   A����   A���(   <���A���   A���(   A��1�   <���A��Dj   A��1�   A��j(   <ò�A��|�   A��j(   A�좨   <��A��j   A�좨   A���(   <�U/A����   A���(   A���   <��A��&j   A���   A��L(   <�vA��^�   A��L(   A��   <��JA��j   A��   A��(   <�\�A����   A��(   A����   <�LlA��j   A����   A��.(   <�?1A��@�   A��.(   A��f�   <�b�A��yj   A��f�   A���(   <���A����   A���(   A��ר   <���A���j   A��ר   A��(   <�~�A��"�   A��(   A��H�   <�X�A��[j   A��H�   A���(   <���A����   A���(   A����   <��6A���j   A����   A���(   <�|MA���   A���(   A� *�   <�VA� =j   A� *�   A�c(   <���A�u�   A�c(   A���   <��$A��j   A���   A��(   <�W�A���   A��(   A��   <�`A�j   A��   A�E(   <���A�W�   A�E(   A�}�   <�;A��j   A�}�   A��(   <�^oA���   A��(   A�	�   <���A�
j   A�	�   A�'(   <�3A�9�   A�'(   A�_�   <���A�rj   A�_�   A��(   <��>A���   A��(   A�Ш   <��BA��j   A�Ш   A�	(   <�3gA��   A�	(   A�A�   <���A�Tj   A�A�   A�z(   <���A���   A�z(   A���   <���A��j   A���   A��(   <�eA���   A��(   A�#�   <��uA�6j   A�#�   A�\(   <��A�n�   A�\(   A���   <�w�A��j   A���   A��(   <�=�A���   A��(   A��   <�H�A�j   A��   A�>(   <�?�A�P�   A�>(   A�v�   <���A��j   A�v�   A��(   <���A���   A��(   A��   <¤�A��j   A��   A�! (   <�ܦA�!2�   A�! (   A�"X�   <�A�"kj   A�"X�   A�#�(   <�`kA�#��   A�#�(   A�$ɨ   <��&A�$�j   A�$ɨ   A�&(   <���A�&�   A�&(   A�':�   <��aA�'Mj   A�':�   A�(s(   <��7A�(��   A�(s(   A�)��   <�o�A�)�j   A�)��   A�*�(   <Ê"A�*��   A�*�(   A�,�   <�8�A�,/j   A�,�   A�-U(   <�U�A�-g�   A�-U(   A�.��   <�dkA�.�j   A�.��   A�/�(   <�^�A�/��   A�/�(   A�0��   <�9�A�1j   A�0��   A�27(   <��hA�2I�   A�27(   A�3o�   <�<�A�3�j   A�3o�   A�4�(   <�6�A�4��   A�4�(   A�5�   <�zA�5�j   A�5�   A�7(   <�U�A�7+�   A�7(   A�8Q�   <�G�A�8dj   A�8Q�   A�9�(   <�k�A�9��   A�9�(   A�:¨   <���A�:�j   A�:¨   A�;�(   <��pA�<�   A�;�(   A�=3�   <���A�=Fj   A�=3�   A�>l(   <��A�>~�   A�>l(   A�?��   <�NXA�?�j   A�?��   A�@�(   <���A�@��   A�@�(   A�B�   <�?A�B(j   A�B�   A�CN(   <�OA�C`�   A�CN(   A�D��   <�H�A�D�j   A�D��   A�E�(   <�lPA�E��   A�E�(   A�F��   <�z$A�G
j   A�F��   A�H0(   <�[UA�HB�   A�H0(   A�Ih�   <�3QA�I{j   A�Ih�   A�J�(   <��pA�J��   A�J�(   A�K٨   <�m�A�K�j   A�K٨   A�M(   <�]�A�M$�   A�M(   A�NJ�   <���A�N]j   A�NJ�   A�O�(   <�+�A�O��   A�O�(   A�P��   <�REA�P�j   A�P��   A�Q�(   <�W�A�R�   A�Q�(   A�S,�   <�-TA�S?j   A�S,�   A�Te(   <�>+A�Tw�   A�Te(   A�U��   <��A�U�j   A�U��   A�V�(   <���A�V��   A�V�(   A�X�   <���A�X!j   A�X�   A�YG(   <�o�A�YY�   A�YG(   A�Z�   <�,xA�Z�j   A�Z�   A�[�(   <�%A�[��   A�[�(   A�\�   <�{A�]j   A�\�   A�^)(   <���A�^;�   A�^)(   A�_a�   <��#A�_tj   A�_a�   A�`�(   <�. A�`��   A�`�(   A�aҨ   <��A�a�j   A�aҨ   A�c(   <�M A�c�   A�c(   A�dC�   <���A�dVj   A�dC�   A�e|(   <�`�A�e��   A�e|(   A�f��   <��CA�f�j   A�f��   A�g�(   <�|�A�g��   A�g�(   A�i%�   <�<�A�i8j   A�i%�   A�j^(   <���A�jp�   A�j^(   A�k��   <��"A�k�j   A�k��   A�l�(   <��/A�l��   A�l�(   A�n�   <�M�A�nj   A�n�   A�o@(   <öA�oR�   A�o@(   A�px�   <���A�p�j   A�px�   A�q�(   <�$AA�q��   A�q�(   A�r�   <�ZA�r�j   A�r�   A�t"(   <�
�A�t4�   A�t"(   A�uZ�   <�1�A�umj   A�uZ�   A�v�(   <ˣ�A�v��   A�v�(   A�w˨   <�)�A�w�j   A�w˨   A�y(   <��A�y�   A�y(   A�z<�   <�.�A�zOj   A�z<�   A�{u(   <�6�A�{��   A�{u(   A�|��   <���A�|�j   A�|��   A�}�(   <�)CA�}��   A�}�(   A��   <��EA�1j   A��   A��W(   <�ŪA��i�   A��W(   A����   <��A���j   A����   A���(   <��VA����   A���(   A�� �   <��A��j   A�� �   A��9(   <�}A��K�   A��9(   A��q�   <���A���j   A��q�   A���(   <�X(A����   A���(   A���   <�m�A���j   A���   A��(   <��yA��-�   A��(   A��S�   <�ҘA��fj   A��S�   A���(   <�Y�A����   A���(   A��Ĩ   <�9A���j   A��Ĩ   A���(   <�|"A���   A���(   A��5�   <�A�A��Hj   A��5�   A��n(   <��A����   A��n(   A����   <�m�A���j   A����   A���(   <���A����   A���(   A���   <���A��*j   A���   A��P(   <��!A��b�   A��P(   A����   <�^�A���j   A����   A���(   <»!A����   A���(   A����   <�.sA��j   A����   A��2(   <�jwA��D�   A��2(   A��j�   <���A��}j   A��j�   A���(   <���A����   A���(   A��ۨ   <�J�A���j   A��ۨ   A��(   <��RA��&�   A��(   A��L�   <�^�A��_j   A��L�   A���(   <�B8A����   A���(   A����   <��A���j   A����   A���(   <�5XA���   A���(   A��.�   <�)A��Aj   A��.�   A��g(   <�=�A��y�   A��g(   A����   <�ITA���j   A����   A���(   <�QWA����   A���(   A���   <�t9A��#j   A���   A��I(   <�swA��[�   A��I(   A����   <͙EA���j   A����   A���(   <�̓A����   A���(   A���   <��*A��j   A���   A��+(   <�i�A��=�   A��+(   A��c�   <�`qA��vj   A��c�   A���(   <���A����   A���(   A��Ԩ   <��0A���j   A��Ԩ   A��(   <�dA���   A��(   A��E�   <˶#A��Xj   A��E�   A��~(   <�N	A����   A��~(   A����   <�S�A���j   A����   A���(   <�X*A���   A���(   A��'�   <���A��:j   A��'�   A��`(   <�c�A��r�   A��`(   A����   <��A���j   A����   A���(   <�6�A����   A���(   A��	�   <��A��j   A��	�   A��B(   <`A��T�   A��B(   A��z�   <�!�A�Íj   A��z�   A�ĳ(   <���A����   A�ĳ(   A���   <��A���j   A���   A��$(   <��*A��6�   A��$(   A��\�   <��A��oj   A��\�   A�ɕ(   <��XA�ɧ�   A�ɕ(   A��ͨ   <�_PA���j   A��ͨ   A��(   <ƞ�A���   A��(   A��>�   <�;MA��Qj   A��>�   A��w(   <��PA�Ή�   A��w(   A�ϯ�   <��A���j   A�ϯ�   A���(   <��A����   A���(   A�� �   <��eA��3j   A�� �   A��Y(   <�t�A��k�   A��Y(   A�ԑ�   <��/A�Ԥj   A�ԑ�   A���(   <¸xA����   A���(   A���   <íyA��j   A���   A��;(   <ˆ�A��M�   A��;(   A��s�   <�@�A�نj   A��s�   A�ڬ(   <�\�A�ھ�   A�ڬ(   A���   <��A���j   A���   A��(   <�ΘA��/�   A��(   A��U�   <��A��hj   A��U�   A�ߎ(   <�.�A�ߠ�   A�ߎ(   A��ƨ   <�)zA���j   A��ƨ   A���(   <�<}A���   A���(   A��7�   <��TA��Jj   A��7�   A��p(   <О�A���   A��p(   A�娨   <ӫ�A��j   A�娨   A���(   <ɏAA����   A���(   A���   <ź�A��,j   A���   A��R(   <ȣ?A��d�   A��R(   A�ꊨ   <� A��j   A�ꊨ   A���(   <��A����   A���(   A����   <�A��j   A����   A��4(   <ܻ�A��F�   A��4(   A��l�   <��vA��j   A��l�   A��(   <�ӌA���   A��(   A��ݨ   <�1BA���j   A��ݨ   A��(   <�K,A��(�   A��(   A��N�   <˦rA��aj   A��N�   A���(   <�p�A����   A���(   A����   <�A���j   A����   A���(   <��DA��
�   A���(   A��0�   <�B�A��Cj   A��0�   A��i(   <�m�A��{�   A��i(   A����   <�yA���j   A����   A���(   <�6�A����   A���(   A���   <�G�A��%j   A���   A��K(   <ˑJA��]�   A��K(   A� ��   <�O�A� �j   A� ��   A��(   <ӟ|A���   A��(   A���   <��>A�j   A���   A�-(   <�g6A�?�   A�-(   A�e�   <՝�A�xj   A�e�   A��(   <�gA���   A��(   A�֨   <�ªA��j   A�֨   A�	(   <�0�A�	!�   A�	(   A�
G�   <��sA�
Zj   A�
G�   A��(   <ُRA���   A��(   A���   <�g(A��j   A���   A��(   <��zA��   A��(   A�)�   <��QA�<j   A�)�   A�b(   <րNA�t�   A�b(   A���   <�p�A��j   A���   A��(   <�g